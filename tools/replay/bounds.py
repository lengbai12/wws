#!/usr/bin/env python3
"""Exact binomial confidence bounds and rejection-operator calibration.

Clopper-Pearson bounds are computed from the regularized incomplete beta
function, so no third-party statistics package is required. A bound is used
instead of an observed rate because a small sample can report zero errors while
the true error rate is still high.

The methods are game-independent: they consume counts only.
"""
from __future__ import annotations

import math

_MAX_ITERATIONS = 300
_EPSILON = 3.0e-16
_TINY = 1.0e-300


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITERATIONS + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPSILON:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), the CDF of a Beta(a, b) distribution."""
    if not (a > 0.0 and b > 0.0):
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def beta_quantile(p: float, a: float, b: float) -> float:
    """Inverse of :func:`regularized_incomplete_beta` by bisection."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be within [0, 1]")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if regularized_incomplete_beta(a, b, middle) < p:
            low = middle
        else:
            high = middle
        if high - low < 1.0e-14:
            break
    return 0.5 * (low + high)


def clopper_pearson(k: int, n: int, level: float = 0.95, sides: int = 2) -> tuple[float, float]:
    """Exact binomial interval for ``k`` successes out of ``n`` trials."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be within [0, n]")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be within (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    alpha = (1.0 - level) / sides
    lower = 0.0 if k == 0 else beta_quantile(alpha, k, n - k + 1)
    upper = 1.0 if k == n else beta_quantile(1.0 - alpha, k + 1, n - k)
    return lower, upper


def upper_bound(k: int, n: int, level: float = 0.95) -> float:
    """One-sided upper bound on the rate, used to bound unseen errors."""
    return clopper_pearson(k, n, level, sides=1)[1]


def calibrate_rejection(rows: list[dict], max_error_fraction: float,
                        level: float = 0.95) -> dict:
    """Largest score-ordered prefix whose error fraction can be bounded.

    ``rows`` holds ``{"score": float, "error": bool}`` and is evaluated in
    descending score order. The largest prefix is reported, not the first
    acceptable one: an exact bound is wide at small counts, so a prefix scan that
    stops at the first wide row would reject every prefix and select nothing.
    """
    if not rows:
        raise ValueError("rows must not be empty")
    if not 0.0 <= max_error_fraction <= 1.0:
        raise ValueError("max_error_fraction must be within [0, 1]")
    ordered = sorted(rows, key=lambda row: -float(row["score"]))
    cumulative = []
    running = 0
    for row in ordered:
        running += int(bool(row["error"]))
        cumulative.append(running)

    accepted = 0
    bound = 1.0
    for size in range(len(ordered), 0, -1):
        candidate = upper_bound(cumulative[size - 1], size, level)
        if candidate <= max_error_fraction:
            accepted, bound = size, candidate
            break
    rejected_at = accepted + 1 if accepted < len(ordered) else None
    return {
        "accepted": accepted, "errors": cumulative[accepted - 1] if accepted else 0,
        "bound": bound, "rejected_at": rejected_at,
        "rejected_score": ordered[rejected_at - 1]["score"] if rejected_at else None,
        "max_error_fraction": max_error_fraction, "level": level,
        "total": len(ordered),
    }

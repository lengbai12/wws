#!/usr/bin/env python3
"""Smoothness of the reticle's approach to the target, measured and enforced.

The servo only records `error_px` per step (target minus reticle), so the quantity
we can actually judge is the error trajectory: a smooth approach shrinks the
error monotonically, a jerky one overshoots and a shaking one alternates sign
without getting closer.

Everything here is a pure function of that series, so the numbers in a report can
be re-derived from the recorded steps instead of trusted.
"""

DEFAULTS = dict(
    near_px=60.0,        # "arrived" radius for overshoot accounting
    jump_px=400.0,       # one step so large the view was thrown, not aimed
    reversal_limit=3,    # sign flips of the step delta inside the window
    window=8,            # steps considered for the oscillation test
    monotonic_floor=0.6,  # share of steps that must reduce the error
)


def _norm(error):
    return (error[0] ** 2 + error[1] ** 2) ** 0.5


def analyse(errors, config=None):
    """Metrics for one engagement's error series (each entry [dx, dy] or None)."""
    cfg = dict(DEFAULTS)
    cfg.update(config or {})
    series = [e for e in errors if e is not None]
    out = dict(steps=len(errors), valid=len(series), smoothness='unknown',
               reason=None, reversals=[0, 0], monotonic_fraction=None,
               overshoot_px=None, max_step_change_px=None, tolerance_steps=None)
    if len(series) < 2:
        out['reason'] = 'too_few_steps'
        return out

    norms = [_norm(e) for e in series]
    deltas = [[series[i][axis] - series[i - 1][axis] for axis in (0, 1)]
              for i in range(1, len(series))]

    reversals = [0, 0]
    for axis in (0, 1):
        signs = [1 if d[axis] > 0 else (-1 if d[axis] < 0 else 0) for d in deltas]
        signs = [s for s in signs if s]
        reversals[axis] = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    out['reversals'] = reversals

    improving = sum(1 for a, b in zip(norms, norms[1:]) if b < a)
    out['monotonic_fraction'] = round(improving / max(1, len(norms) - 1), 3)

    out['max_step_change_px'] = round(max(_norm(d) for d in deltas), 1)

    # Overshoot: the worst error seen after first coming inside near_px.
    inside = next((i for i, n in enumerate(norms) if n <= cfg['near_px']), None)
    if inside is not None:
        out['overshoot_px'] = round(max(norms[inside:]), 1)
        out['tolerance_steps'] = sum(1 for n in norms[inside:] if n <= cfg['near_px'])

    window = reversals[0] >= cfg['reversal_limit'] or reversals[1] >= cfg['reversal_limit']
    # Oscillation is about the swings, not about whether the error shrinks:
    # measured 2026-10-23, a series can close 400 -> 380 -> 370 while reversing
    # direction every single step, and that is exactly the shaking camera the
    # operator complained about. A lone huge step (few reversals) stays 'jerky'.
    if window and out['max_step_change_px'] > cfg['near_px']:
        out['smoothness'], out['reason'] = 'oscillating', 'sign_flips_without_progress'
    elif out['max_step_change_px'] > cfg['jump_px']:
        out['smoothness'], out['reason'] = 'jerky', 'single_step_too_large'
    elif out['monotonic_fraction'] < cfg['monotonic_floor']:
        out['smoothness'], out['reason'] = 'jerky', 'not_monotonic'
    else:
        out['smoothness'] = 'smooth'
    return out


def guard(errors, config=None):
    """Live decision for the servo: may the loop keep stepping?

    Returns (action, reason) where action is ``continue``, ``damp`` or ``stop``.
    A loop that alternates direction without getting closer is shaking the
    camera, which looks broken and wastes frames; damping first and stopping
    after a second warning keeps the aim usable.
    """
    cfg = dict(DEFAULTS)
    cfg.update(config or {})
    series = [e for e in errors if e is not None]
    if len(series) < cfg['window']:
        return 'continue', None
    recent = series[-cfg['window']:]
    metrics = analyse(recent, cfg)
    verdict = metrics['smoothness']
    if verdict == 'oscillating':
        return 'stop', metrics['reason']
    if verdict == 'jerky':
        return 'damp', metrics['reason']
    return 'continue', None

#!/usr/bin/env python3
"""Operator budget: <=10 s aim, then 20 s of clicking with no re-aiming."""
import argparse
import unittest
from pathlib import Path

import dynamic_shot_run as run


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeProbe:
    def __init__(self):
        self.calls = []

    def x(self, *args):
        self.calls.append(tuple(str(a) for a in args))
        return ''


class BurstTests(unittest.TestCase):
    def test_twenty_seconds_at_half_a_second_is_forty_clicks(self):
        clock, probe = FakeClock(), FakeProbe()
        clicks = run.burst_fire(probe, 20.0, 0.5, clock=clock, sleep=clock.sleep)
        self.assertEqual(clicks, 40)
        self.assertEqual(len(probe.calls), 80)
        self.assertEqual(probe.calls[:2], [('mousedown', '1'), ('mouseup', '1')])
        # Nothing but clicks: no aiming input may appear inside the burst.
        self.assertEqual({call[0] for call in probe.calls}, {'mousedown', 'mouseup'})

    def test_the_burst_stops_at_the_window(self):
        clock, probe = FakeClock(), FakeProbe()
        run.burst_fire(probe, 3.0, 0.5, clock=clock, sleep=clock.sleep)
        self.assertLess(clock.now, 3.0 + 0.5)
        self.assertEqual(len(probe.calls), 12)

    def test_a_click_always_pairs_a_release(self):
        clock, probe = FakeClock(), FakeProbe()
        run.burst_fire(probe, 2.0, 0.25, clock=clock, sleep=clock.sleep)
        downs = [c for c in probe.calls if c[0] == 'mousedown']
        ups = [c for c in probe.calls if c[0] == 'mouseup']
        self.assertEqual(len(downs), len(ups))

    def test_invalid_windows_are_rejected(self):
        probe = FakeProbe()
        for seconds, interval in ((0.0, 0.5), (-1.0, 0.5), (20.0, 0.0), (20.0, -0.5)):
            with self.subTest(seconds=seconds, interval=interval):
                with self.assertRaises(ValueError):
                    run.burst_fire(probe, seconds, interval)


class AimBudgetTests(unittest.TestCase):
    def argv(self, **kwargs):
        base = dict(python='py', aim_seconds=10.0, execute=True)
        base.update(kwargs)
        return run.servo_argv(argparse.Namespace(**base), Path('/tmp/run-01'))

    def test_the_aim_phase_carries_the_ten_second_budget(self):
        # The budget is measured inside the servo from the search start, so it
        # travels as --aim-budget-s; --seconds only bounds the process.
        argv = self.argv()
        self.assertEqual(argv[argv.index('--aim-budget-s') + 1], '10.0')
        self.assertEqual(argv[argv.index('--seconds') + 1], '90')

    def test_a_shorter_budget_is_honoured(self):
        argv = self.argv(aim_seconds=6.0)
        self.assertEqual(argv[argv.index('--aim-budget-s') + 1], '6.0')

    def test_the_step_cap_is_bounded(self):
        self.assertEqual(self.argv()[self.argv().index('--max-steps') + 1], '16')

    def test_fire_is_only_requested_when_executing(self):
        self.assertIn('--fire', self.argv())
        self.assertNotIn('--fire', self.argv(execute=False))


if __name__ == '__main__':
    unittest.main()


class BurstSamplingTests(unittest.TestCase):
    """Clicks are not shots: the burst samples the reload state to count them."""

    def test_the_sampler_runs_on_its_own_period(self):
        clock, probe = FakeClock(), FakeProbe()
        seen = []
        run.burst_fire(probe, 20.0, 0.5, clock=clock, sleep=clock.sleep,
                       sample=lambda: seen.append(clock()), sample_every=2.0)
        # The window is half-open: samples land at 2,4,...,18 inside 20 s.
        self.assertEqual(len(seen), 9)
        self.assertEqual(seen[0], 2.0)
        self.assertEqual(seen[-1], 18.0)

    def test_no_sampler_means_pure_clicks(self):
        clock, probe = FakeClock(), FakeProbe()
        clicks = run.burst_fire(probe, 4.0, 0.5, clock=clock, sleep=clock.sleep)
        self.assertEqual(clicks, 8)

    def test_a_zero_sampling_period_is_rejected(self):
        clock, probe = FakeClock(), FakeProbe()
        with self.assertRaises(ValueError):
            run.burst_fire(probe, 4.0, 0.5, clock=clock, sleep=clock.sleep,
                           sample=lambda: None, sample_every=0)


class SalvoCounterTests(unittest.TestCase):
    def test_ready_to_reloading_counts_one_salvo(self):
        counter = run.SalvoCounter()
        for ready in (True, True, False, False, True, True, False, True):
            counter.observe(ready)
        self.assertEqual(counter.salvos, 2)

    def test_unknown_readings_never_count(self):
        counter = run.SalvoCounter()
        for ready in (None, True, None, False, None):
            counter.observe(ready)
        self.assertEqual(counter.salvos, 1)
        self.assertEqual(counter.unknown, 3)

    def test_unreadable_after_ready_counts_at_the_second_sample(self):
        # Live frames: ready True, then the reload bars read unknown, never False.
        counter = run.SalvoCounter()
        for ready in (True, None, None, None):
            counter.observe(ready)
        self.assertEqual(counter.salvos, 1)

    def test_a_lone_unknown_never_counts(self):
        counter = run.SalvoCounter()
        for ready in (True, None, True, None, True):
            counter.observe(ready)
        self.assertEqual(counter.salvos, 0)

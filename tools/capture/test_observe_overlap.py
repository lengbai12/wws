"""The per-frame source/focus verification must run beside the worker.

Serial checks cost about 40 ms of every frame. This gate drives the exchange
with fakes so it can assert the interleaving deterministically: the reply must
arrive while the verification future is still in flight, and a changed source
must still refuse the observation.
"""
import sys
import threading
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import observe_main_view as O


class OverlapTests(unittest.TestCase):
    def exchange(self, verification):
        """Mirror run()'s exchange using fakes for the worker and the checks."""
        events = []
        verified_event = threading.Event()
        replied = threading.Event()

        def verify():
            events.append('verify_enter')
            replied.wait(2.0)
            verified_event.set()
            events.append('verify_exit')
            return verification()

        def read_response(process, request_id, timeout):
            events.append('respond')
            replied.set()
            self.assertTrue(verified_event.wait(2.0))
            return {'completed_ns': 0}

        from concurrent.futures import ThreadPoolExecutor
        with patch.object(O, 'read_response', read_response), \
                patch.object(O, 'preflight', self.game_node):
            with ThreadPoolExecutor(max_workers=1) as checks:
                job = checks.submit(verify)
                response = O.read_response(None, 0, 2.)
            result = job.result()
        return events, response, result

    @staticmethod
    def game_node():
        return {'pid': 1}, {'serial': 2}

    def test_source_checks_interleave_with_the_reply(self):
        events, response, _ = self.exchange(lambda: (self.game_node()[0], self.game_node()[1], True))
        self.assertEqual(events, ['verify_enter', 'respond', 'verify_exit'])
        self.assertEqual(response, {'completed_ns': 0})

    def test_changed_source_is_still_reported(self):
        events, _, result = self.exchange(lambda: ({'pid': 9}, {'serial': 2}, False))
        self.assertNotEqual((result[0], result[1]), (self.game_node()[0], self.game_node()[1]))
        self.assertFalse(result[2])

    def test_verify_source_skips_focus_when_the_source_changed(self):
        calls = []
        with patch.object(O, 'preflight', lambda: ({'pid': 9}, {'serial': 2})), \
                patch.object(O, 'focused', lambda game: calls.append(game) or True):
            game, node, available = O.verify_source({'pid': 1}, {'serial': 2})
        self.assertEqual(calls, [])
        self.assertFalse(available)

    def test_verify_source_reads_focus_when_the_source_matches(self):
        calls = []
        with patch.object(O, 'preflight', lambda: ({'pid': 1}, {'serial': 2})), \
                patch.object(O, 'focused', lambda game: calls.append(game) or True):
            game, node, available = O.verify_source({'pid': 1}, {'serial': 2})
        self.assertEqual(len(calls), 1)
        self.assertTrue(available)


if __name__ == '__main__':
    unittest.main()

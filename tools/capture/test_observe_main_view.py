"""Live observer transport deadlines and source delivery-age boundaries."""
import os
from pathlib import Path
import subprocess
from threading import Event
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from observe_main_view import (delivery_age_ns, read_line, read_response, receive_observation,
                               send, preflight, focused, receipt_continuity, track_continuity)


class LiveObserverTests(unittest.TestCase):
    def test_focus_accepts_current_game_pid(self):
        with patch('observe_main_view.subprocess.check_output', return_value=b'20\n'):
            self.assertTrue(focused(dict(pid=20, display=':2')))

    def test_focus_fails_closed_for_wrong_pid_invalid_output_or_query_failure(self):
        game = dict(pid=20, display=':2')
        for output in (b'21\n', b'', b'20\n21\n'):
            with self.subTest(output=output), \
                 patch('observe_main_view.subprocess.check_output', return_value=output):
                self.assertFalse(focused(game))
        for error in (subprocess.TimeoutExpired('xdotool', 1),
                      subprocess.CalledProcessError(1, 'xdotool')):
            with self.subTest(error=error), \
                 patch('observe_main_view.subprocess.check_output', side_effect=error):
                self.assertFalse(focused(game))

    def test_preflight_preserves_verified_source_and_game_ancestry(self):
        game = dict(pid=20, display=':2', start_ticks='123')
        node = dict(id=3, name='gamescope', serial=40, process_id=10, sec_uid=os.getuid())
        with patch('capture_host.resolve_target', return_value=node), \
             patch('capture_host.game_pids', return_value={20}), \
             patch('probe_aim_input.process_identity', return_value=game), \
             patch('probe_aim_input.ancestor_pids', return_value={20, 10}):
            actual_game, actual_node = preflight()
        self.assertEqual(game, actual_game)
        self.assertEqual({k: node[k] for k in ('id', 'name', 'serial', 'process_id')}, actual_node)

    def test_preflight_rejects_ambiguous_games_or_unrelated_source(self):
        node = dict(id=3, name='gamescope', serial=40, process_id=10)
        for pids, ancestors in ((set(), {10}), ({20, 30}, {10}), ({20}, {20, 99})):
            with self.subTest(pids=pids, ancestors=ancestors), \
                 patch('capture_host.resolve_target', return_value=node), \
                 patch('capture_host.game_pids', return_value=pids), \
                 patch('probe_aim_input.process_identity', return_value=dict(pid=20)), \
                 patch('probe_aim_input.ancestor_pids', return_value=ancestors), \
                 self.assertRaises(RuntimeError):
                preflight()

    def test_preflight_does_not_hide_resolver_rejection(self):
        from gamescope_record import CaptureError
        with patch('capture_host.resolve_target', side_effect=CaptureError('ambiguous source')), \
             self.assertRaises(CaptureError):
            preflight()

    def test_preflight_overlaps_live_reads_without_reusing_results(self):
        game_started, source_started = Event(), Event()

        def games():
            game_started.set()
            if not source_started.wait(1):
                raise AssertionError('source lookup did not overlap game lookup')
            return {20}

        def source(*args):
            if not game_started.wait(1):
                raise AssertionError('game lookup did not overlap source lookup')
            source_started.set()
            return dict(id=3, name='gamescope', serial=40, process_id=10)

        identities = [dict(pid=20, display=':2', start_ticks=start)
                      for start in ('123', '456')]
        with patch('capture_host.resolve_target', side_effect=source) as resolve, \
             patch('capture_host.game_pids', side_effect=games) as enumerate_games, \
             patch('probe_aim_input.process_identity', side_effect=identities) as identity, \
             patch('probe_aim_input.ancestor_pids', return_value={20, 10}) as ancestors:
            first, _ = preflight()
            game_started.clear()
            source_started.clear()
            second, _ = preflight()
        self.assertEqual('123', first['start_ticks'])
        self.assertEqual('456', second['start_ticks'])
        for check in (resolve, enumerate_games, identity, ancestors):
            self.assertEqual(2, check.call_count)

    def test_preflight_propagates_live_process_failures(self):
        from probe_aim_input import ProbeError
        node = dict(id=3, name='gamescope', serial=40, process_id=10)
        for helper, error in (('process_identity', ProbeError('identity changed')),
                              ('ancestor_pids', ProcessLookupError('ancestor exited'))):
            with self.subTest(helper=helper), \
                 patch('capture_host.resolve_target', return_value=node), \
                 patch('capture_host.game_pids', return_value={20}), \
                 patch('probe_aim_input.process_identity', return_value=dict(pid=20)), \
                 patch(f'probe_aim_input.{helper}', side_effect=error), \
                 self.assertRaises(type(error)):
                preflight()
    def test_receipt_continuity_requires_coverage_and_advancing_source(self):
        def row(seq, received, expiry, pts, **changes):
            observed = dict(fresh=True, captured_ns=received-10,
                            valid_until_ns=expiry, epoch=1, seq=seq)
            observed.update(changes)
            return dict(source_pts_ns=pts, receiver=dict(received_ns=received),
                        observation=observed)

        first = row(0, 100, 200, 100)
        covered = row(1, 199, 299, 199)
        report = receipt_continuity([first, covered])
        self.assertTrue(report['entire_sequence_dense_and_fresh'])
        self.assertEqual(2, report['longest_dense_fresh_run_frames'])
        for second in (row(1, 200, 300, 200), row(1, 201, 301, 201),
                       row(1, 199, 299, 100), row(1, 199, 299, 199, epoch=2),
                       row(2, 199, 299, 199), row(1, 199, None, 199, fresh=False),
                       row(1, 199, 199, 199)):
            with self.subTest(second=second):
                report = receipt_continuity([first, second])
                self.assertFalse(report['entire_sequence_dense_and_fresh'])
                self.assertEqual(1, report['longest_dense_fresh_run_frames'])

    def test_receipt_continuity_reports_gap_and_restarts_after_rejection(self):
        rows = [dict(source_pts_ns=pts, receiver=dict(received_ns=received),
                     observation=dict(fresh=fresh, captured_ns=received-10 if fresh else None,
                         valid_until_ns=expiry, epoch=1, seq=seq))
                for seq, pts, received, expiry, fresh in
                ((0, 100, 100, 150, True), (1, 200, 220, 300, True),
                 (2, 250, 260, None, False), (3, 280, 290, 400, True),
                 (4, 300, 320, 500, True))]
        report = receipt_continuity(rows)
        self.assertEqual(70, report['pairs'][0]['uncovered_ns'])
        self.assertIsNone(report['pairs'][2]['uncovered_ns'])
        self.assertEqual(4, report['fresh_frames'])
        self.assertEqual(2, report['longest_dense_fresh_run_frames'])
        self.assertFalse(report['entire_sequence_dense_and_fresh'])
        self.assertFalse(receipt_continuity([])['entire_sequence_dense_and_fresh'])
    def test_track_continuity_requires_dense_unique_evidence(self):
        def row(seq, received, pts, evidence, *, fresh=True, epoch=1, expiry=None):
            expiry = received + 100 if expiry is None and fresh else expiry
            return dict(source_pts_ns=pts, receiver=dict(received_ns=received),
                        track_evidence=evidence,
                        observation=dict(fresh=fresh, captured_ns=received - 1 if fresh else None,
                            valid_until_ns=expiry, epoch=epoch, seq=seq))
        tracked = [dict(kind='target', track_id=7, state='tracked')]
        report = track_continuity([row(0, 100, 100, tracked),
                                   row(1, 199, 199, tracked),
                                   row(2, 298, 298, [dict(kind='target', track_id=None,
                                                          state='ambiguous')]),
                                   row(3, 397, 397, tracked),
                                   row(4, 496, 496, tracked, epoch=2)])
        self.assertEqual(2, report['target']['longest_run_frames'])
        self.assertEqual(0, report['marker']['longest_run_frames'])

    def test_track_continuity_uses_raw_jsonl_and_excludes_hud_dangers(self):
        rows = []
        for seq, state in enumerate(('tentative', 'tracked', 'tracked')):
            received = 100 + seq * 50
            rows.append(dict(source_pts_ns=received, receiver=dict(received_ns=received),
                observation=dict(fresh=True, captured_ns=received-1,
                    valid_until_ns=received+100, epoch=1, seq=seq,
                    view=dict(detections=[dict(track_id=7, track_state=state)],
                              markers=[dict(track_id=None, track_state='observed',
                                  marker_family='central_dangers')]))))
        report = track_continuity(rows)
        self.assertEqual(3, report['target']['longest_run_frames'])
        self.assertEqual(0, report['marker']['track_count'])
        self.assertNotIn('track_evidence', rows[0])
        self.assertNotIn('track_evidence', rows[0]['observation'])
        import copy
        for change in ('missing', 'ambiguous', 'stale', 'deadline', 'seq', 'epoch',
                       'source_gap', 'receipt_gap', 'tentative'):
            with self.subTest(change=change):
                broken = copy.deepcopy(rows)
                middle = broken[1]
                observed = middle['observation']
                if change == 'missing':
                    observed['view']['detections'] = []
                elif change in ('ambiguous', 'tentative'):
                    observed['view']['detections'][0]['track_state'] = change
                elif change == 'stale':
                    observed.update(fresh=False, captured_ns=None, valid_until_ns=None)
                elif change == 'deadline':
                    observed['valid_until_ns'] = middle['receiver']['received_ns']
                elif change in ('seq', 'epoch'):
                    observed[change] += 3
                elif change == 'source_gap':
                    middle['source_pts_ns'] += 500_000_000
                else:
                    middle['receiver']['received_ns'] += 500_000_000
                self.assertLess(track_continuity(broken)['target']['longest_run_frames'], 3)

    def test_delivery_age_includes_queue_and_refresh(self):
        self.assertEqual(50, delivery_age_ns(100, 130, 20))
        for values in ((-1, 100, 1), (101, 100, 1), (True, 100, 1), (1, 2, -1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                delivery_age_ns(*values)
    def response(self, *, intent_expiry=170):
        return dict(completed_ns=120,
            observation=dict(fresh=True, reason='fresh', captured_ns=100,
                valid_until_ns=200, age_ns=25, view={'detections': [1]},
                hud={'battle': {'active': True}}, target={'id': 1},
                lock={'confirmed': True}, camera={'stable': True},
                mouse_response={'status': 'measured'}, aim_settled=True,
                fire_authorized=True),
            intent=dict(action='fire_once', fire_authorized=True, target=1,
                aim_pixel=[1, 2], relative_mouse=[1, 2],
                valid_until_ns=intent_expiry, release_buttons=False))

    def test_receiver_counts_transport_age_without_extending_expiry(self):
        response = receive_observation(self.response(), 150, focused=True, source_available=True)
        self.assertTrue(response['observation']['fresh'])
        self.assertEqual(55, response['observation']['age_ns'])
        self.assertEqual(200, response['observation']['valid_until_ns'])
        self.assertEqual('fire_once', response['intent']['action'])
        self.assertEqual(170, response['intent']['valid_until_ns'])

    def test_worker_fresh_response_is_cleared_at_receiver_deadline(self):
        for now in (200, 201):
            response = receive_observation(self.response(), now, focused=True, source_available=True)
            self.assertTrue(response['receiver']['worker_fresh'])
            self.assertFalse(response['observation']['fresh'])
            for field in ('view', 'hud', 'target', 'valid_until_ns'):
                self.assertIsNone(response['observation'][field])
            self.assertFalse(response['observation']['lock']['confirmed'])
            self.assertFalse(response['intent']['fire_authorized'])
            self.assertIsNone(response['intent']['relative_mouse'])
            self.assertEqual('receiver_observation_expired', response['intent']['reason'])
            self.assertTrue(response['intent']['release_buttons'])

    def test_receiver_rechecks_focus_and_source_after_inference(self):
        for focus, source, reason in ((False, True, 'focus_lost'),
                                      (True, False, 'source_lost')):
            response = receive_observation(self.response(), 150, focused=focus, source_available=source)
            self.assertFalse(response['observation']['fresh'])
            self.assertIsNone(response['observation']['view'])
            self.assertEqual('revoke', response['intent']['action'])
            self.assertEqual(reason, response['intent']['reason'])

    def test_intent_can_expire_before_visual_evidence(self):
        response = receive_observation(self.response(), 170, focused=True, source_available=True)
        self.assertTrue(response['observation']['fresh'])
        self.assertFalse(response['intent']['fire_authorized'])
        self.assertEqual('receiver_intent_expired', response['intent']['reason'])
        self.assertEqual(170, response['intent']['valid_until_ns'])

    def test_receiver_does_not_reanimate_worker_rejection(self):
        response = self.response()
        response['observation'].update(fresh=False, reason='processing_expired',
            captured_ns=None, valid_until_ns=None, age_ns=None, view=None)
        receive_observation(response, 150, focused=True, source_available=True)
        self.assertFalse(response['observation']['fresh'])
        self.assertEqual('processing_expired', response['intent']['reason'])
        self.assertEqual('revoke', response['intent']['action'])

    def test_context_lost_during_inference_revokes_before_publication(self):
        import observe_main_view as observer
        pinned_game, pinned_node = dict(pid=20), dict(serial=40)
        for loss in ('focus', 'source'):
            with self.subTest(loss=loss):
                state = dict(reply_received=False)

                def reply(*args):
                    state['reply_received'] = True
                    return self.response()

                def context():
                    self.assertTrue(state['reply_received'])
                    return (dict(pid=21) if loss == 'source' else pinned_game,
                            pinned_node)

                with patch.object(observer, 'read_response', side_effect=reply), \
                     patch.object(observer, 'preflight', side_effect=context), \
                     patch.object(observer, 'focused', return_value=False) as focus, \
                     patch.object(observer.time, 'monotonic_ns', side_effect=(130, 140, 150)):
                    result, responded, verified, received, available = observer.receive_verified_response(
                        None, 0, pinned_game, pinned_node)
                self.assertEqual((130, 140, 150), (responded, verified, received))
                self.assertEqual(loss != 'source', available)
                self.assertEqual(0 if loss == 'source' else 1, focus.call_count)
                self.assertFalse(result['observation']['fresh'])
                self.assertIsNone(result['observation']['view'])
                self.assertEqual('revoke', result['intent']['action'])
                self.assertEqual(f'{loss}_lost', result['intent']['reason'])

    def test_post_response_context_check_time_consumes_original_lease(self):
        import observe_main_view as observer
        game, node = dict(pid=20), dict(serial=40)
        with patch.object(observer, 'read_response', return_value=self.response()), \
             patch.object(observer, 'verify_source', return_value=(game, node, True)), \
             patch.object(observer.time, 'monotonic_ns', side_effect=(150, 199, 200)):
            result, *_ = observer.receive_verified_response(None, 0, game, node)
        self.assertFalse(result['observation']['fresh'])
        self.assertEqual('receiver_observation_expired', result['intent']['reason'])
        self.assertEqual(150, result['receiver']['context_check_started_ns'])
        self.assertEqual(199, result['receiver']['context_check_completed_ns'])

    def test_post_response_context_check_preserves_remaining_lease(self):
        import observe_main_view as observer
        game, node = dict(pid=20), dict(serial=40)
        with patch.object(observer, 'read_response', return_value=self.response()), \
             patch.object(observer, 'verify_source', return_value=(game, node, True)), \
             patch.object(observer.time, 'monotonic_ns', side_effect=(130, 140, 150)):
            result, *_ = observer.receive_verified_response(None, 0, game, node)
        self.assertTrue(result['observation']['fresh'])
        self.assertEqual(200, result['observation']['valid_until_ns'])
        self.assertEqual(170, result['intent']['valid_until_ns'])
        self.assertEqual('fire_once', result['intent']['action'])

    def test_partial_response_without_newline_has_deadline(self):
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b'{"partial":')
            with os.fdopen(read_fd, 'rb', buffering=0) as source:
                with self.assertRaises(TimeoutError):
                    read_line(SimpleNamespace(stdout=source), .02)
        finally:
            os.close(write_fd)

    def test_timeout_preserves_partial_frame_and_correlates_shutdown(self):
        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(read_fd, 'rb', buffering=0) as source:
                process = SimpleNamespace(stdout=source)
                os.write(write_fd, b'{"request_id":17,"intent":')
                with self.assertRaises(TimeoutError):
                    read_line(process, .02)
                os.write(write_fd, b'"hold"}\n{"request_id":"shutdown","intent":"revoke"}\n')
                stopped = read_response(process, 'shutdown', .2)
                self.assertEqual('revoke', stopped['intent'])
                self.assertEqual(bytearray(), process._response_buffer)

        finally:
            os.close(write_fd)

    def test_unresponsive_worker_cannot_block_large_payload(self):
        read_fd, write_fd = os.pipe()
        try:
            os.set_blocking(write_fd, False)
            with os.fdopen(write_fd, 'wb', buffering=0) as destination:
                with self.assertRaises(TimeoutError):
                    send(SimpleNamespace(stdin=destination), {'bytes': 1000000}, b'x'*1000000, .02)
        finally:
            os.close(read_fd)

    def test_fresh_process_reads_fragmented_payload_then_revokes(self):
        # A real consumer, no display or capture services required. Black RGB
        # has no battle HUD and must not publish targets or leave a lease.
        script = Path(__file__).with_name('observe_main_view.py')
        python = script.parents[2]/'.venv/bin/python'
        process = subprocess.Popen([str(python), '-B', str(script), '--worker'],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        try:
            self.assertEqual({'ready': True}, read_line(process, 15))
            import time
            payload = bytes(1600*2560*3)
            stamp = dict(epoch=0, seq=0, captured_ns=time.monotonic_ns(), age_bound_ns=0)
            os.set_blocking(process.stdin.fileno(), False)
            send(process, dict(bytes=len(payload), stamp=stamp, focused=True, request_id=0), payload)
            response = read_response(process, 0, 5)
            self.assertIsNone(response['observation']['view'])
            self.assertEqual({'payload_read', 'observe', 'snapshot_policy'},
                             set(response['worker_timings_ms']))
            self.assertTrue(all(value >= 0 for value in response['worker_timings_ms'].values()))
            self.assertFalse(response['intent']['fire_authorized'])
            send(process, {'revoke': 'source_lost', 'request_id': 'stop'})
            stopped = read_response(process, 'stop', 2)
            self.assertEqual('source_lost', stopped['observation']['reason'])
            self.assertFalse(stopped['observation']['fresh'])
            self.assertEqual('source_lost', stopped['intent']['reason'])
            self.assertIsNone(stopped['intent']['relative_mouse'])
            self.assertTrue(stopped['intent']['release_buttons'])
            self.assertIsNone(stopped['observation']['lock'].get('target_track_id'))
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            process.stdout.close()


if __name__ == '__main__':
    unittest.main()

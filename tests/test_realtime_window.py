from __future__ import annotations

import unittest

from moss_transcribe_diarize.realtime.window import WindowPolicy


def _policy(**kwargs) -> WindowPolicy:
    params = {"window": 20.0, "hop": 5.0, "min_first_window": 8.0}
    params.update(kwargs)
    return WindowPolicy(**params)


class WindowPolicyTest(unittest.TestCase):
    def test_waits_for_min_first_window(self):
        policy = _policy()

        self.assertFalse(policy.decide(total_seconds=7.9, last_run_sec=None).should_run)
        decision = policy.decide(total_seconds=8.0, last_run_sec=None)
        self.assertTrue(decision.should_run)
        self.assertEqual(decision.start_sec, 0.0)
        self.assertEqual(decision.end_sec, 8.0)

    def test_first_window_start_is_clamped_to_zero(self):
        policy = _policy(min_first_window=30.0)

        decision = policy.decide(total_seconds=30.0, last_run_sec=None)

        # 先钉住"这次运行真的发生了"：WindowDecision(False) 的 start_sec 默认就是 0.0，
        # 一个"首次运行永不触发"的错误实现也能满足下面那条断言。
        self.assertTrue(decision.should_run)
        self.assertEqual(decision.start_sec, 0.0)

    def test_later_runs_are_throttled_by_hop(self):
        policy = _policy()

        self.assertFalse(policy.decide(total_seconds=32.4, last_run_sec=28.0).should_run)
        self.assertTrue(policy.decide(total_seconds=33.0, last_run_sec=28.0).should_run)

    def test_later_window_is_window_seconds_wide(self):
        policy = _policy()

        decision = policy.decide(total_seconds=60.0, last_run_sec=55.0)

        self.assertEqual(decision.start_sec, 40.0)
        self.assertEqual(decision.end_sec, 60.0)

    def test_later_window_start_is_clamped_at_zero(self):
        policy = _policy(window=40.0, hop=5.0)

        decision = policy.decide(total_seconds=25.0, last_run_sec=20.0)

        self.assertEqual(decision.start_sec, 0.0)
        self.assertEqual(decision.end_sec, 25.0)

    def test_rejects_hop_longer_than_window(self):
        with self.assertRaises(ValueError):
            _policy(window=10.0, hop=11.0)

    def test_rejects_non_positive_min_first_window(self):
        with self.assertRaises(ValueError):
            _policy(min_first_window=0.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from moss_transcribe_diarize.realtime.config import RealtimeConfig


class RealtimeConfigTest(unittest.TestCase):
    def test_defaults_are_valid(self):
        config = RealtimeConfig()

        # 这些默认值是跨任务契约：后续 Task 直接读它们（例如 Task 10 的测试
        # 送 8.0 秒音频并期望触发推理，只在 min_first_window == 8.0 时成立），
        # 所以每个字段都逐个钉死，改动默认值必须让这里失败。
        self.assertEqual(config.window, 20.0)
        self.assertEqual(config.hop, 5.0)
        self.assertEqual(config.tail, 6.0)
        self.assertEqual(config.min_first_window, 8.0)
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.buffer_capacity, 180.0)
        self.assertTrue(config.silence_gate)
        self.assertEqual(config.silence_rms_db, -45.0)
        self.assertEqual(config.silence_frame_ratio, 0.05)
        self.assertEqual(config.speaker_threshold, 0.55)
        self.assertEqual(config.min_segment_sec, 0.4)
        self.assertIsNone(config.max_new_tokens)
        self.assertEqual(config.poll_interval, 0.5)
        self.assertEqual(config.max_consecutive_failures, 3)

    def test_effective_max_new_tokens_scales_with_window(self):
        self.assertEqual(RealtimeConfig(window=20.0).effective_max_new_tokens(), 1020)
        self.assertEqual(RealtimeConfig(window=40.0).effective_max_new_tokens(), 2040)

    def test_effective_max_new_tokens_has_a_floor(self):
        self.assertEqual(
            RealtimeConfig(window=1.0, hop=1.0, tail=0.0).effective_max_new_tokens(), 256
        )

    def test_explicit_max_new_tokens_wins(self):
        config = RealtimeConfig(window=20.0, max_new_tokens=77)

        self.assertEqual(config.effective_max_new_tokens(), 77)

    def test_rejects_hop_longer_than_window(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(window=10.0, hop=20.0)

    def test_rejects_tail_at_or_past_window(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(window=10.0, hop=5.0, tail=10.0)

    def test_rejects_negative_tail(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(tail=-1.0)

    def test_rejects_buffer_too_small_for_window(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(window=20.0, hop=5.0, buffer_capacity=25.0)

    def test_rejects_out_of_range_similarity_threshold(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(speaker_threshold=1.5)

    def test_rejects_negative_similarity_threshold(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(speaker_threshold=-0.1)

    def test_rejects_out_of_range_silence_ratio(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(silence_frame_ratio=0.0)

    def test_rejects_silence_ratio_above_one(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(silence_frame_ratio=1.5)

    def test_rejects_non_positive_sample_rate(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(sample_rate=0)

    def test_rejects_non_positive_window(self):
        # ``window <= 0`` 被 ``hop > window`` 覆盖（hop 默认 5.0），所以裸的
        # assertRaises 无法区分：删掉这条规则后仍会因 hop 检查抛错，测试照样通过。
        # 钉住消息，使本测试在规则被删时确实失败。
        with self.assertRaises(ValueError) as ctx:
            RealtimeConfig(window=0.0)

        self.assertIn("window must be positive", str(ctx.exception))

    def test_rejects_non_positive_hop(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(hop=0.0)

    def test_rejects_non_positive_min_first_window(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(min_first_window=0.0)

    def test_rejects_silence_rms_db_below_floor(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(silence_rms_db=-121.0)

    def test_rejects_silence_rms_db_above_ceiling(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(silence_rms_db=1.0)

    def test_rejects_negative_min_segment_sec(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(min_segment_sec=-0.1)

    def test_rejects_non_positive_max_new_tokens(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(max_new_tokens=0)

    def test_rejects_non_positive_poll_interval(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(poll_interval=0.0)

    def test_rejects_non_positive_max_consecutive_failures(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(max_consecutive_failures=0)

    def test_config_is_frozen(self):
        config = RealtimeConfig()

        with self.assertRaises(Exception):
            config.window = 30.0


if __name__ == "__main__":
    unittest.main()

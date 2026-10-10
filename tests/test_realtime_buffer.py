from __future__ import annotations

import threading
import unittest

import numpy as np

from moss_transcribe_diarize.realtime.buffer import AudioRingBuffer


def _ramp(start: float, count: int) -> np.ndarray:
    return np.arange(start, start + count, dtype=np.float32)


class AudioRingBufferTest(unittest.TestCase):
    def test_slice_returns_appended_audio(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 5))

        np.testing.assert_allclose(buffer.slice(0.0, 0.5), _ramp(0, 5))

    def test_slice_uses_seconds_not_samples(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 10))

        np.testing.assert_allclose(buffer.slice(0.2, 0.5), _ramp(2, 3))

    def test_total_and_available_seconds(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 6))

        self.assertAlmostEqual(buffer.total_seconds, 0.6)
        self.assertAlmostEqual(buffer.available_seconds, 0.6)

    def test_wrap_preserves_logical_order(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 7))
        buffer.append(_ramp(7, 6))  # 总量 13，容量 10，最旧 3 个样本被丢弃

        self.assertAlmostEqual(buffer.available_seconds, 1.0)
        np.testing.assert_allclose(buffer.slice(0.3, 1.3), _ramp(3, 10))

    def test_slice_before_retained_head_returns_none(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 13))

        self.assertIsNone(buffer.slice(0.0, 0.5))
        self.assertIsNotNone(buffer.slice(0.3, 1.0))

    def test_slice_clamps_at_total(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 5))

        np.testing.assert_allclose(buffer.slice(0.0, 99.0), _ramp(0, 5))

    def test_slice_with_inverted_range_returns_none(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 5))

        self.assertIsNone(buffer.slice(0.4, 0.2))

    def test_slice_past_total_after_head_returns_none(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 5))

        self.assertIsNone(buffer.slice(0.6, 0.9))

    def test_oversized_append_keeps_only_the_tail(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 25))

        self.assertAlmostEqual(buffer.total_seconds, 2.5)
        self.assertAlmostEqual(buffer.available_seconds, 1.0)
        np.testing.assert_allclose(buffer.slice(1.5, 2.5), _ramp(15, 10))

    def test_empty_append_is_a_noop(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(np.zeros(0, dtype=np.float32))

        self.assertEqual(buffer.total_samples, 0)

    def test_append_accepts_non_float32_input(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append([0.0, 0.5, 1.0])

        np.testing.assert_allclose(buffer.slice(0.0, 0.3), np.array([0.0, 0.5, 1.0], dtype=np.float32))

    def test_concurrent_append_and_slice(self):
        buffer = AudioRingBuffer(capacity_seconds=2.0, sample_rate=100)
        errors: list[BaseException] = []

        def writer():
            try:
                for _ in range(200):
                    buffer.append(np.ones(50, dtype=np.float32))
            except BaseException as exc:  # pragma: no cover - 只在失败时触发
                errors.append(exc)

        def reader():
            try:
                for _ in range(200):
                    window = buffer.slice(0.5, 1.5)
                    if window is not None:
                        self.assertEqual(window.dtype, np.float32)
            except BaseException as exc:  # pragma: no cover - 只在失败时触发
                errors.append(exc)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()

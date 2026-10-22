from __future__ import annotations

import unittest

import numpy as np

from moss_transcribe_diarize.realtime.energy import frame_rms_db, is_silent


def _tone(seconds: float, *, amplitude: float = 0.3, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(round(seconds * sample_rate)), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _silence(seconds: float, *, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(round(seconds * sample_rate)), dtype=np.float32)


class FrameRmsDbTest(unittest.TestCase):
    def test_silence_is_very_low(self):
        levels = frame_rms_db(_silence(0.1), 16000)

        self.assertTrue(np.all(levels < -100.0))

    def test_tone_is_loud(self):
        levels = frame_rms_db(_tone(0.2, amplitude=0.5), 16000)

        self.assertTrue(np.all(levels > -20.0))

    def test_frame_count_matches_the_hop(self):
        # 0.1 秒音频 = 1600 样本；20ms 窗 = 320，10ms 移 = 160
        # count = 1 + (1600 - 320) // 160 = 9
        levels = frame_rms_db(_silence(0.1), 16000, frame_ms=20.0, hop_ms=10.0)

        self.assertEqual(levels.size, 9)

    def test_short_audio_still_yields_one_frame(self):
        levels = frame_rms_db(_silence(0.001), 16000)

        self.assertEqual(levels.size, 1)

    def test_empty_audio_yields_one_frame(self):
        levels = frame_rms_db(_silence(0.0), 16000)

        self.assertEqual(levels.size, 1)


class IsSilentTest(unittest.TestCase):
    def test_pure_silence_is_silent(self):
        self.assertTrue(is_silent(_silence(2.0), 16000, -45.0, 0.05))

    def test_continuous_speech_is_not_silent(self):
        self.assertFalse(is_silent(_tone(2.0), 16000, -45.0, 0.05))

    def test_short_burst_in_a_long_window_is_silent(self):
        # 0.5 秒有声落在 20 秒窗口里，活跃帧占比约 2.5%，远低于 5% 门限。
        # 不要用 1.0 秒——那算出来是 5.0025%，正好压在门限上。
        audio = np.concatenate([_silence(19.5), _tone(0.5)])

        self.assertTrue(is_silent(audio, 16000, -45.0, 0.05))

    def test_enough_speech_in_a_long_window_is_not_silent(self):
        audio = np.concatenate([_silence(14.0), _tone(6.0)])

        self.assertFalse(is_silent(audio, 16000, -45.0, 0.05))

    def test_empty_audio_is_silent(self):
        self.assertTrue(is_silent(_silence(0.0), 16000, -45.0, 0.05))


if __name__ == "__main__":
    unittest.main()

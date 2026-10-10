from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from moss_transcribe_diarize.realtime.transcriber import (
    HfWindowTranscriber,
    VllmWindowTranscriber,
    WindowTranscriber,
    token_budget,
)


class _Result:
    def __init__(self, text: str):
        self.text = text


class FakeRunner:
    def __init__(self, text: str = "[0.5][S01]你好[1.5]"):
        self.text = text
        self.calls: list[dict] = []

    def transcribe(self, audio_path, *, prompt, max_new_tokens, decoding):
        self.calls.append(
            {
                "path": Path(audio_path),
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "decoding": decoding,
            }
        )
        return _Result(self.text)


class HfWindowTranscriberTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.scratch = Path(self._tmp.name) / "scratch"

    def test_is_a_window_transcriber(self):
        transcriber = HfWindowTranscriber(FakeRunner(), self.scratch)

        self.assertIsInstance(transcriber, WindowTranscriber)

    def test_returns_runner_text(self):
        transcriber = HfWindowTranscriber(FakeRunner("结果"), self.scratch)

        self.assertEqual(
            transcriber.transcribe_window(np.zeros(1600, dtype=np.float32), prompt="p"),
            "结果",
        )

    def test_creates_the_scratch_directory(self):
        HfWindowTranscriber(FakeRunner(), self.scratch)

        self.assertTrue(self.scratch.is_dir())

    def test_writes_a_readable_16k_wav(self):
        runner = FakeRunner()
        transcriber = HfWindowTranscriber(runner, self.scratch)
        audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)

        transcriber.transcribe_window(audio, prompt="p")

        written, rate = sf.read(str(runner.calls[0]["path"]), dtype="float32")
        self.assertEqual(rate, 16000)
        self.assertEqual(written.shape, (16000,))
        self.assertAlmostEqual(float(np.max(np.abs(written - audio))), 0.0, places=4)

    def test_overwrites_the_same_file_each_call(self):
        runner = FakeRunner()
        transcriber = HfWindowTranscriber(runner, self.scratch)
        transcriber.transcribe_window(np.zeros(1600, dtype=np.float32), prompt="p")
        first_size = runner.calls[0]["path"].stat().st_size

        transcriber.transcribe_window(np.zeros(800, dtype=np.float32), prompt="p")

        self.assertEqual(runner.calls[0]["path"], runner.calls[1]["path"])
        self.assertLess(runner.calls[1]["path"].stat().st_size, first_size)

    def test_forwards_prompt_and_budget(self):
        runner = FakeRunner()
        transcriber = HfWindowTranscriber(runner, self.scratch, max_new_tokens=777, decoding="sample")

        transcriber.transcribe_window(np.zeros(1600, dtype=np.float32), prompt="自定义")

        self.assertEqual(runner.calls[0]["prompt"], "自定义")
        self.assertEqual(runner.calls[0]["max_new_tokens"], 777)
        self.assertEqual(runner.calls[0]["decoding"], "sample")

    def test_accepts_an_empty_window_without_crashing(self):
        """空窗口在管线里应被上游拦住，但这里也不能因为写文件而崩。"""
        runner = FakeRunner("")
        transcriber = HfWindowTranscriber(runner, self.scratch)

        self.assertEqual(
            transcriber.transcribe_window(np.zeros(0, dtype=np.float32), prompt="p"), ""
        )


class TokenBudgetTest(unittest.TestCase):
    def test_scales_with_the_actual_audio(self):
        # 20 秒 -> 20*51 = 1020
        self.assertEqual(token_budget(np.zeros(20 * 16000, dtype=np.float32), 16000), 1020)

    def test_a_long_forced_window_gets_a_bigger_budget(self):
        # 覆盖性保证会让被迫放行的窗口长于 config.window（实测到过 100 秒），
        # 按 window 定死的预算会把它们的尾部截掉。
        self.assertEqual(token_budget(np.zeros(60 * 16000, dtype=np.float32), 16000), 3060)

    def test_floor_applies_to_short_windows(self):
        self.assertEqual(token_budget(np.zeros(16000, dtype=np.float32), 16000, floor=1024), 1024)

    def test_cap_wins_when_given(self):
        self.assertEqual(
            token_budget(np.zeros(60 * 16000, dtype=np.float32), 16000, cap=999), 999
        )

    def test_never_returns_less_than_the_minimum(self):
        self.assertEqual(token_budget(np.zeros(16, dtype=np.float32), 16000), 256)


class HfWindowTranscriberBudgetTest(unittest.TestCase):
    """沿用本文件已有的 FakeRunner。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.scratch = Path(self._tmp.name) / "scratch"

    def test_budget_follows_the_window_it_is_given(self):
        runner = FakeRunner()
        transcriber = HfWindowTranscriber(runner, self.scratch, token_budget_floor=256)

        transcriber.transcribe_window(np.zeros(20 * 16000, dtype=np.float32), prompt="p")

        self.assertEqual(runner.calls[-1]["max_new_tokens"], 1020)

    def test_explicit_max_new_tokens_still_caps(self):
        runner = FakeRunner()
        transcriber = HfWindowTranscriber(runner, self.scratch, max_new_tokens=777)

        transcriber.transcribe_window(np.zeros(60 * 16000, dtype=np.float32), prompt="p")

        self.assertEqual(runner.calls[-1]["max_new_tokens"], 777)


class VllmWindowTranscriberTest(unittest.TestCase):
    def test_is_a_window_transcriber(self):
        transcriber = VllmWindowTranscriber(base_url="http://x", model="m")
        self.assertIsInstance(transcriber, WindowTranscriber)

    def test_window_is_encoded_and_posted_without_touching_torch(self):
        sent: list[dict] = []

        def fake_post(**kwargs):
            sent.append(kwargs)
            return {"text": "[0.5][S01]你好[1.5]", "usage": {"completion_tokens": 9}}

        transcriber = VllmWindowTranscriber(
            base_url="http://x", model="m", token_budget_floor=256, post=fake_post
        )

        text = transcriber.transcribe_window(np.zeros(20 * 16000, dtype=np.float32), prompt="p")

        self.assertEqual(text, "[0.5][S01]你好[1.5]")
        self.assertEqual(sent[0]["model"], "m")
        self.assertEqual(sent[0]["prompt"], "p")
        self.assertEqual(sent[0]["max_new_tokens"], 1020)
        self.assertTrue(sent[0]["file_bytes"].startswith(b"RIFF"))

    def test_module_imports_without_torch(self):
        code = (
            "import sys\n"
            "for name in ('torch', 'transformers', 'moss_transcribe_diarize.app.model_runner'):\n"
            "    sys.modules[name] = None\n"
            "import moss_transcribe_diarize.realtime.transcriber as t\n"
            "assert hasattr(t, 'VllmWindowTranscriber')\n"
            "print('IMPORT_OK')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertIn("IMPORT_OK", result.stdout, result.stderr)


if __name__ == "__main__":
    unittest.main()

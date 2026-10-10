import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import moss_transcribe_diarize.app.vllm_runner as vllm_runner_module
from moss_transcribe_diarize.app.openai_audio_client import transcriptions_url
from moss_transcribe_diarize.app.vllm_runner import VllmRunner


class VllmRunnerTest(unittest.TestCase):
    def test_transcribe_posts_openai_compatible_audio_transcription_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sample.wav"
            sf.write(audio_path, np.zeros(1600, dtype=np.float32), 16000)
            runner = VllmRunner(base_url="http://vllm.test:8000", model="moss-vllm", api_key="secret")
            calls = []
            original = vllm_runner_module.transcribe_bytes

            def fake_transcribe_bytes(**kwargs):
                calls.append(kwargs)
                return {
                    "text": "[0][S01]hello[1.5]",
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                }

            vllm_runner_module.transcribe_bytes = fake_transcribe_bytes
            self.addCleanup(setattr, vllm_runner_module, "transcribe_bytes", original)
            status = []
            result = runner.transcribe(
                audio_path,
                prompt="transcribe",
                max_new_tokens=128,
                decoding="sample",
                temperature=0.8,
                status_callback=lambda state, progress, tokens=None: status.append((state, progress, tokens)),
            )

            self.assertEqual(result.text, "[0][S01]hello[1.5]")
            self.assertEqual(result.prompt_len, 11)
            self.assertEqual(result.generated_tokens, 7)
            payload = calls[0]
            self.assertEqual(payload["base_url"], "http://vllm.test:8000")
            self.assertEqual(payload["model"], "moss-vllm")
            self.assertEqual(payload["prompt"], "transcribe")
            self.assertEqual(payload["api_key"], "secret")
            self.assertEqual(payload["filename"], "audio.wav")
            self.assertEqual(payload["max_new_tokens"], 128)
            self.assertEqual(payload["decoding"], "sample")
            self.assertEqual(payload["temperature"], 0.8)
            self.assertTrue(payload["file_bytes"].startswith(b"RIFF"))
            self.assertEqual(status[-1], ("transcribing", 0.85, 7))

    def test_transcription_url_accepts_v1_or_full_endpoint(self):
        self.assertEqual(
            transcriptions_url("http://host:8000/v1"),
            "http://host:8000/v1/audio/transcriptions",
        )
        self.assertEqual(
            transcriptions_url("http://host:8000/v1/audio/transcriptions"),
            "http://host:8000/v1/audio/transcriptions",
        )


if __name__ == "__main__":
    unittest.main()

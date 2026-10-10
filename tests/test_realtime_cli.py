from __future__ import annotations

import unittest
from pathlib import Path

from moss_transcribe_diarize.app.realtime_cli import build_embedder, parse_args


class ParseArgsTest(unittest.TestCase):
    def test_defaults(self):
        args = parse_args([])

        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 7870)
        self.assertEqual(args.window, 20.0)
        self.assertEqual(args.hop, 5.0)
        self.assertEqual(args.tail, 6.0)
        self.assertEqual(args.backend, "hf")

    def test_vllm_backend_takes_a_base_url(self):
        args = parse_args(["--backend", "vllm", "--vllm-base-url", "http://host:8000"])
        self.assertEqual(args.vllm_base_url, "http://host:8000")

    def test_no_speaker_disables_the_embedder_flag(self):
        self.assertFalse(parse_args(["--no-speaker"]).speaker)
        self.assertTrue(parse_args([]).speaker)

    def test_no_silence_gate_and_no_record(self):
        args = parse_args(["--no-silence-gate", "--no-record"])
        self.assertFalse(args.silence_gate)
        self.assertFalse(args.record)

    def test_help_mentions_the_loopback_default_so_nobody_exposes_it_by_accident(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buffer):
            parse_args(["--help"])
        self.assertIn("127.0.0.1", buffer.getvalue())


class BuildEmbedderTest(unittest.TestCase):
    def test_no_speaker_gives_none(self):
        self.assertIsNone(build_embedder(parse_args(["--no-speaker"])))

    def test_speaker_model_path_is_honoured_without_downloading(self):
        args = parse_args(["--speaker-model", "some/path.onnx"])
        # 只验证它把路径传下去了；真正构造需要 onnxruntime 与那个文件
        self.assertEqual(args.speaker_model, "some/path.onnx")


class BuildConfigTest(unittest.TestCase):
    def test_flags_reach_the_config(self):
        from moss_transcribe_diarize.app.realtime_cli import build_config

        config = build_config(parse_args([]))

        self.assertEqual(config.window, 20.0)
        self.assertEqual(config.hop, 5.0)
        self.assertEqual(config.tail, 6.0)
        self.assertIsNone(config.max_new_tokens)

    def test_max_new_tokens_override_reaches_the_config(self):
        from moss_transcribe_diarize.app.realtime_cli import build_config

        self.assertEqual(build_config(parse_args(["--max-new-tokens", "512"])).max_new_tokens, 512)


class BuildTranscriberFactoryTest(unittest.TestCase):
    def test_vllm_without_a_base_url_stops_with_a_message(self):
        from moss_transcribe_diarize.app.realtime_cli import build_config, build_transcriber_factory

        args = parse_args(["--backend", "vllm"])

        with self.assertRaises(SystemExit) as ctx:
            build_transcriber_factory(args, build_config(args))
        self.assertIn("--vllm-base-url", str(ctx.exception))

    def test_vllm_builds_a_transcriber_without_torch(self):
        """`--backend vllm` 下绝不能加载 torch——这是整个轻量部署的前提。"""
        import subprocess
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        code = (
            "import sys\n"
            "for name in ('torch', 'transformers', 'moss_transcribe_diarize.app.model_runner'):\n"
            "    sys.modules[name] = None\n"
            "from moss_transcribe_diarize.app.realtime_cli import build_config, build_transcriber_factory, parse_args\n"
            "args = parse_args(['--backend', 'vllm', '--vllm-base-url', 'http://x'])\n"
            "t = build_transcriber_factory(args, build_config(args))()\n"
            "assert type(t).__name__ == 'VllmWindowTranscriber', type(t)\n"
            "assert sys.modules.get('torch') is None, sys.modules.get('torch')\n"
            "assert sys.modules.get('moss_transcribe_diarize.app.model_runner') is None\n"
            "print('IMPORT_OK')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=repo)
        self.assertIn("IMPORT_OK", result.stdout, result.stderr)


class HelpOnANonUtf8ConsoleTest(unittest.TestCase):
    """英文 Windows 的默认编码是 cp1252，而这个 CLI 的帮助与提示都是中文。"""

    def test_help_exits_cleanly_when_stdout_cannot_encode_chinese(self):
        import os
        import subprocess
        import sys

        env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
        result = subprocess.run(
            [sys.executable, "-m", "moss_transcribe_diarize.app.realtime_cli", "--help"],
            capture_output=True, text=True, env=env,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("UnicodeEncodeError", result.stderr)


def _inline_source(statements: list[str]) -> str:
    """把几行源码拼成 python -c 的入参；写成真正的换行，别用转义。"""
    return chr(10).join(statements) + chr(10)


class BuildRetranscriberTest(unittest.TestCase):
    def test_vllm_without_a_base_url_stops_with_a_message(self):
        from moss_transcribe_diarize.app.realtime_cli import build_config, build_retranscriber

        args = parse_args(["--backend", "vllm"])

        with self.assertRaises(SystemExit) as ctx:
            build_retranscriber(args, build_config(args))
        self.assertIn("--vllm-base-url", str(ctx.exception))

    def test_vllm_retranscriber_needs_no_torch(self):
        import subprocess
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        source = _inline_source([
            "import sys",
            "for name in ('torch', 'transformers', 'moss_transcribe_diarize.app.model_runner'):",
            "    sys.modules[name] = None",
            "from moss_transcribe_diarize.app.realtime_cli import build_config, build_retranscriber, parse_args",
            "args = parse_args(['--backend', 'vllm', '--vllm-base-url', 'http://x'])",
            "r = build_retranscriber(args, build_config(args))",
            "assert callable(r)",
            "assert sys.modules.get('torch') is None, sys.modules.get('torch')",
            "print('IMPORT_OK')",
        ])
        result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, cwd=repo)
        self.assertIn("IMPORT_OK", result.stdout, result.stderr)

    def test_hf_retranscriber_does_not_load_the_model_until_it_is_used(self):
        """它要占一份额外的显存，而多数人从不点那个按钮。"""
        import subprocess
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        source = _inline_source([
            "import sys",
            "from moss_transcribe_diarize.app.realtime_cli import build_config, build_retranscriber, parse_args",
            "args = parse_args(['--backend', 'hf', '--model', 'runs/model'])",
            "r = build_retranscriber(args, build_config(args))",
            "assert callable(r)",
            "print('NO_LOAD' if 'torch.nn' not in sys.modules else 'LOADED')",
        ])
        result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, cwd=repo)
        self.assertIn("NO_LOAD", result.stdout, result.stderr)


class WholeRecordingBudgetTest(unittest.TestCase):
    """整段录音的预算按**实际时长**算——按 window 定死会截掉绝大部分内容。"""

    def _write(self, tmp, seconds: float):
        import numpy as np
        import soundfile as sf

        path = Path(tmp) / f"{seconds}.wav"
        sf.write(str(path), np.zeros(int(seconds * 16000), dtype=np.float32), 16000)
        return path

    def test_the_budget_grows_with_the_recording_length(self):
        import tempfile

        from moss_transcribe_diarize.app.realtime_cli import (
            _budget_for_a_whole_recording,
            build_config,
        )

        config = build_config(parse_args([]))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_budget_for_a_whole_recording(self._write(tmp, 30), config, None), 1530)
            self.assertEqual(_budget_for_a_whole_recording(self._write(tmp, 600), config, None), 30600)

    def test_an_explicit_override_wins(self):
        import tempfile

        from moss_transcribe_diarize.app.realtime_cli import (
            _budget_for_a_whole_recording,
            build_config,
        )

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                _budget_for_a_whole_recording(self._write(tmp, 600), build_config(parse_args([])), 512),
                512,
            )


if __name__ == "__main__":
    unittest.main()

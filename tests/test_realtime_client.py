from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from realtime_client import chunk_pcm, wav_to_pcm16k  # noqa: E402


class ChunkPcmTest(unittest.TestCase):
    def test_splits_into_fixed_sized_chunks(self):
        chunks = list(chunk_pcm(np.arange(10, dtype=np.float32), 4))
        self.assertEqual([c.size for c in chunks], [4, 4, 2])

    def test_empty_input_gives_no_chunks(self):
        self.assertEqual(list(chunk_pcm(np.zeros(0, dtype=np.float32), 4)), [])

    def test_chunk_is_float32_little_endian_ready(self):
        chunk = next(chunk_pcm(np.array([1.5], dtype=np.float32), 1))
        self.assertEqual(chunk.dtype, np.float32)
        self.assertEqual(len(chunk.tobytes()), 4)


class WavLoadingTest(unittest.TestCase):
    def test_resamples_to_16k_mono(self):
        import soundfile as sf
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.wav"
            sf.write(str(path), np.zeros(48000, dtype=np.float32), 48000)

            pcm = wav_to_pcm16k(path)

            self.assertEqual(pcm.shape, (16000,))
            self.assertEqual(pcm.dtype, np.float32)

    def test_mixes_a_stereo_file_down_to_mono(self):
        import soundfile as sf
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stereo.wav"
            left = np.full(16000, 0.5, dtype=np.float32)
            right = np.full(16000, -0.5, dtype=np.float32)
            sf.write(str(path), np.stack([left, right], axis=1), 16000)

            pcm = wav_to_pcm16k(path)

            self.assertEqual(pcm.shape, (16000,))
            np.testing.assert_allclose(pcm, 0.0, atol=1e-6)


class DrainedPacingTest(unittest.TestCase):
    def _drain_in_a_loop(self, samples, *, chunk_samples, realtime):
        import asyncio
        from unittest import mock

        from realtime_client import _drain

        async def collect():
            with mock.patch("asyncio.sleep", new=mock.AsyncMock()) as sleep:
                chunks = [chunk async for chunk in _drain(
                    np.zeros(samples, dtype=np.float32),
                    chunk_samples=chunk_samples, realtime=realtime, rate=16000)]
            return chunks, [call.args[0] for call in sleep.await_args_list]

        return asyncio.run(collect())

    def test_fast_mode_does_not_sleep(self):
        chunks, awaited = self._drain_in_a_loop(8000, chunk_samples=1600, realtime=False)

        self.assertEqual(len(chunks), 5)
        self.assertEqual(awaited, [])

    def test_realtime_mode_paces_at_the_chunk_duration(self):
        chunks, awaited = self._drain_in_a_loop(4800, chunk_samples=1600, realtime=True)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(awaited, [0.1, 0.1, 0.1])

    def test_pacing_yields_to_the_event_loop(self):
        """用 time.sleep 会把事件循环堵死：收不到服务端事件，也回不了它的 ping。"""
        import inspect

        from realtime_client import _drain

        self.assertTrue(inspect.isasyncgenfunction(_drain))


class ServerClosesTheConnectionTest(unittest.TestCase):
    """服务端处理完 stop 就关连接。那是收尾，不该让客户端抛出去。"""

    def test_run_survives_a_close_right_after_stop(self):
        import asyncio
        import tempfile

        import soundfile as sf
        from websockets.exceptions import ConnectionClosed

        from realtime_client import run

        class FakeSocket:
            def __init__(self):
                self.sent = []

            async def send(self, data):
                self.sent.append(data)

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise ConnectionClosed(None, None)

        class FakeConnect:
            async def __aenter__(self):
                return FakeSocket()

            async def __aexit__(self, *exc):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.wav"
            sf.write(str(path), np.zeros(16000, dtype=np.float32), 16000)

            stats = asyncio.run(
                run("ws://x", path, realtime=False, connect=lambda url: FakeConnect(), tail_wait=0.0)
            )

        self.assertEqual(stats, {"committed": 0, "errors": 0, "events": 0})


class TeardownEventsAreDeliveredTest(unittest.TestCase):
    """收尾那几段（往往正是会议最后几句）在 stop **之后**才由服务端发出。

    等一个固定秒数会漏掉它们：盘上的 transcript.jsonl 是完整的，而终端上看到的少了
    一截——"看到的"与"存下来的"对不上是最难查的一类不一致。
    """

    def test_events_arriving_after_stop_are_printed_and_counted(self):
        import asyncio
        import json
        import tempfile

        import soundfile as sf
        from websockets.exceptions import ConnectionClosed

        from realtime_client import run

        class FakeSocket:
            def __init__(self):
                self.sent = []
                self.stopped = asyncio.Event()
                self.queue = []

            async def send(self, data):
                self.sent.append(data)
                if isinstance(data, str) and json.loads(data).get("type") == "stop":
                    self.queue.append(json.dumps({
                        "type": "committed",
                        "segments": [{"start": 60.0, "end": 62.1, "speaker": "S02",
                                      "speaker_name": "S02", "text": "会后我发纪要"}],
                    }))
                    self.stopped.set()

            def __aiter__(self):
                return self

            async def __anext__(self):
                await self.stopped.wait()
                if self.queue:
                    return self.queue.pop(0)
                raise ConnectionClosed(None, None)

        class FakeConnect:
            def __init__(self):
                self.socket = FakeSocket()

            async def __aenter__(self):
                return self.socket

            async def __aexit__(self, *exc):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.wav"
            sf.write(str(path), np.zeros(1600, dtype=np.float32), 16000)

            stats = asyncio.run(run(
                "ws://x", path, realtime=False,
                connect=lambda url: FakeConnect(), tail_wait=0.0, stop_timeout=5.0,
            ))

        self.assertEqual(stats["committed"], 1)


class NonUtf8ConsoleTest(unittest.TestCase):
    """输出的每一行都是中文，而英文 Windows 的控制台是 cp1252。"""

    def test_help_survives_a_non_utf8_stdout(self):
        import os
        import subprocess
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
        result = subprocess.run(
            [sys.executable, "scripts/realtime_client.py", "--help"],
            capture_output=True, text=True, env=env, cwd=repo,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("UnicodeEncodeError", result.stderr)


if __name__ == "__main__":
    unittest.main()
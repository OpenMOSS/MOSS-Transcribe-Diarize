from __future__ import annotations

import io
import json
import subprocess
import sys
import unittest
import urllib.request

import numpy as np
import soundfile as sf

from moss_transcribe_diarize.app.openai_audio_client import (
    build_multipart_body,
    consume_sse_transcription,
    encode_wav_bytes,
    extract_transcription_text,
    transcribe_bytes,
    transcriptions_url,
)


class TorchFreeTest(unittest.TestCase):
    def test_module_imports_without_torch(self):
        """这个模块是实时路径的地基，它一旦拖 torch，整个服务的轻量部署就没了。"""
        code = (
            "import sys\n"
            "for name in ('torch', 'transformers', 'moss_transcribe_diarize.app.model_runner'):\n"
            "    sys.modules[name] = None\n"
            "import moss_transcribe_diarize.app.openai_audio_client as c\n"
            "assert hasattr(c, 'transcribe_bytes')\n"
            "print('IMPORT_OK')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertIn("IMPORT_OK", result.stdout, result.stderr)


class TranscriptionsUrlTest(unittest.TestCase):
    def test_appends_the_path_to_a_bare_host(self):
        self.assertEqual(
            transcriptions_url("http://127.0.0.1:8000"),
            "http://127.0.0.1:8000/v1/audio/transcriptions",
        )

    def test_respects_a_trailing_slash(self):
        self.assertEqual(
            transcriptions_url("http://127.0.0.1:8000/"),
            "http://127.0.0.1:8000/v1/audio/transcriptions",
        )

    def test_keeps_an_explicit_v1(self):
        self.assertEqual(
            transcriptions_url("http://host/v1"),
            "http://host/v1/audio/transcriptions",
        )

    def test_keeps_an_explicit_full_path(self):
        self.assertEqual(
            transcriptions_url("http://host/v1/audio/transcriptions"),
            "http://host/v1/audio/transcriptions",
        )


class EncodeWavBytesTest(unittest.TestCase):
    def test_round_trips_through_soundfile(self):
        pcm = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)

        data = encode_wav_bytes(pcm, 16000)

        self.assertTrue(data.startswith(b"RIFF"))
        back, rate = sf.read(io.BytesIO(data), dtype="float32")
        self.assertEqual(rate, 16000)
        # 16-bit PCM，所以只要求近似（量化误差）
        np.testing.assert_allclose(back, pcm, atol=1e-4)

    def test_accepts_an_empty_window_without_crashing(self):
        data = encode_wav_bytes(np.zeros(0, dtype=np.float32), 16000)
        self.assertTrue(data.startswith(b"RIFF"))


class BuildMultipartBodyTest(unittest.TestCase):
    def test_puts_every_field_in_the_body(self):
        body = build_multipart_body(
            boundary="BOUND",
            fields={"model": "m", "prompt": "p"},
            file_field="file",
            filename="audio.wav",
            content_type="audio/wav",
            file_bytes=b"\x01\x02",
        )

        text = body.decode("latin-1")
        self.assertIn('name="model"', text)
        self.assertIn("m", text)
        self.assertIn('name="prompt"', text)
        self.assertIn('filename="audio.wav"', text)
        self.assertIn("Content-Type: audio/wav", text)
        self.assertTrue(body.endswith(b"--BOUND--\r\n"))

    def test_file_bytes_are_raw_not_escaped(self):
        payload = bytes(range(256))

        body = build_multipart_body(
            boundary="B", fields={}, file_field="file", filename="a.wav",
            content_type="audio/wav", file_bytes=payload,
        )

        self.assertIn(payload, body)


class ExtractTranscriptionTextTest(unittest.TestCase):
    def test_reads_the_text_field(self):
        self.assertEqual(extract_transcription_text({"text": "  你好  "}), "你好")

    def test_missing_text_is_empty_not_none(self):
        self.assertEqual(extract_transcription_text({}), "")
        self.assertEqual(extract_transcription_text({"text": None}), "")


class _FakeResponse:
    """把 SSE 行喂给 consume_sse_transcription，不碰网络。"""

    def __init__(self, lines: list[str]):
        self._lines = [line.encode("utf-8") for line in lines]

    def __iter__(self):
        return iter(self._lines)


class ConsumeSseTest(unittest.TestCase):
    def test_joins_delta_content(self):
        response = _FakeResponse([
            'data: {"choices":[{"delta":{"content":"[0.5][S01]"}}]}',
            'data: {"choices":[{"delta":{"content":"你好[1.5]"}}]}',
            "data: [DONE]",
        ])

        result = consume_sse_transcription(response)

        self.assertEqual(result["text"], "[0.5][S01]你好[1.5]")

    def test_collects_usage_when_present(self):
        response = _FakeResponse([
            'data: {"choices":[{"delta":{"content":"x"}}],"usage":{"prompt_tokens":7,"completion_tokens":9}}',
            "data: [DONE]",
        ])

        result = consume_sse_transcription(response)

        self.assertEqual(result["usage"], {"prompt_tokens": 7, "completion_tokens": 9})

    def test_on_progress_gets_the_completion_token_count(self):
        seen: list[int] = []
        response = _FakeResponse([
            'data: {"choices":[{"delta":{"content":"x"}}],"usage":{"completion_tokens":3}}',
            'data: {"choices":[{"delta":{"content":"y"}}],"usage":{"completion_tokens":4}}',
            "data: [DONE]",
        ])

        consume_sse_transcription(response, on_progress=seen.append)

        self.assertEqual(seen, [3, 4])

    def test_ignores_lines_that_are_not_data(self):
        response = _FakeResponse(["", ": keep-alive", "data: [DONE]"])

        self.assertEqual(consume_sse_transcription(response)["text"], "")


class _CapturedRequest:
    """拦住 urlopen，把发出去的请求记下来，再回一个预置响应。"""

    def __init__(self, body: bytes = b'{"text":"ok"}', content_type: str = "application/json"):
        self.requests: list = []
        self._body = body
        self._content_type = content_type

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        outer = self

        class _Response:
            headers = {"Content-Type": outer._content_type}

            def read(self):
                return outer._body

            def __iter__(self):
                return iter(outer._body.splitlines(keepends=True))

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Response()


class TranscribeBytesTest(unittest.TestCase):
    """``transcribe_bytes`` 是 vLLM 那条路上的全部请求构造，字段一个都不能漂。

    这些断言原先住在 ``tests/test_vllm_runner.py`` 里，抽取之后逻辑搬到了这里，
    断言也跟着搬——留在原处就只是钉一个空壳。
    """

    def setUp(self):
        self._urlopen_original = urllib.request.urlopen

    def _install(self, **kwargs) -> _CapturedRequest:
        captured = _CapturedRequest(**kwargs)
        urllib.request.urlopen = captured
        self.addCleanup(setattr, urllib.request, "urlopen", self._urlopen_original)
        return captured

    def test_builds_the_openai_compatible_payload(self):
        captured = self._install()
        progress: list[int] = []

        result = transcribe_bytes(
            base_url="http://vllm.test:8000/",
            model="moss-vllm",
            prompt="  transcribe  ",
            file_bytes=b"RIFFxxxx",
            api_key="secret",
            max_new_tokens=128,
            decoding="sample",
            temperature=0.8,
            on_progress=progress.append,
        )

        self.assertEqual(result["text"], "ok")
        request = captured.requests[0]
        self.assertEqual(request.full_url, "http://vllm.test:8000/v1/audio/transcriptions")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        body = request.data.decode("latin-1")
        self.assertIn('name="model"', body)
        self.assertIn("moss-vllm", body)
        self.assertIn("transcribe", body)
        self.assertNotIn("  transcribe  ", body)          # prompt 会被 strip
        self.assertIn('name="response_format"', body)
        self.assertIn("json", body)
        self.assertIn('name="stream"', body)
        self.assertIn('name="stream_include_usage"', body)
        self.assertIn('name="stream_continuous_usage_stats"', body)
        self.assertIn('name="max_completion_tokens"', body)
        self.assertIn("128", body)
        self.assertIn('name="temperature"', body)
        self.assertIn("0.8", body)
        self.assertIn('filename="audio.wav"', body)
        self.assertIn("Content-Type: audio/wav", body)
        self.assertIn("RIFFxxxx", body)

    def test_greedy_decoding_pins_the_temperature_to_zero(self):
        captured = self._install()

        transcribe_bytes(
            base_url="http://x", model="m", prompt="p", file_bytes=b"x",
            decoding="greedy", temperature=0.8,
        )

        self.assertIn("0.0", captured.requests[0].data.decode("latin-1"))

    def test_consumes_an_sse_response(self):
        self._install(
            body=b'data: {"choices":[{"delta":{"content":"x"}}]}\ndata: [DONE]\n',
            content_type="text/event-stream",
        )

        result = transcribe_bytes(base_url="http://x", model="m", prompt="p", file_bytes=b"x")

        self.assertEqual(result["text"], "x")

    def test_a_non_json_body_falls_back_to_raw_text(self):
        self._install(body=b"plain words", content_type="text/plain")

        result = transcribe_bytes(base_url="http://x", model="m", prompt="p", file_bytes=b"x")

        self.assertEqual(result["text"], "plain words")


class CheckEndpointUrlTest(unittest.TestCase):
    """探测与转写请求必须拼同一个根，否则会互相矛盾：请求能过、横幅说不可达。"""

    def _probe_url(self, base_url):
        import urllib.request

        from moss_transcribe_diarize.app.openai_audio_client import check_endpoint, transcriptions_url

        seen = []
        original = urllib.request.urlopen

        class _Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout=None):
            seen.append(request.full_url)
            return _Response()

        urllib.request.urlopen = fake_urlopen
        try:
            check_endpoint(base_url)
        finally:
            urllib.request.urlopen = original
        return seen[0], transcriptions_url(base_url)

    def test_a_bare_host_probes_the_v1_models_route(self):
        probed, transcribed = self._probe_url("http://127.0.0.1:8000")

        self.assertEqual(probed, "http://127.0.0.1:8000/v1/models")
        self.assertEqual(transcribed, "http://127.0.0.1:8000/v1/audio/transcriptions")

    def test_an_explicit_v1_is_not_doubled(self):
        probed, transcribed = self._probe_url("http://host:8000/v1")

        self.assertEqual(probed, "http://host:8000/v1/models")
        self.assertEqual(transcribed, "http://host:8000/v1/audio/transcriptions")

    def test_a_custom_path_prefix_is_kept(self):
        probed, transcribed = self._probe_url("http://host/openai")

        self.assertEqual(probed, "http://host/openai/v1/models")
        self.assertEqual(transcribed, "http://host/openai/v1/audio/transcriptions")

    def test_a_full_transcriptions_url_degrades_to_the_same_root(self):
        probed, transcribed = self._probe_url("http://host:8000/v1/audio/transcriptions")

        self.assertEqual(probed, "http://host:8000/v1/models")
        self.assertEqual(transcribed, "http://host:8000/v1/audio/transcriptions")


if __name__ == "__main__":
    unittest.main()

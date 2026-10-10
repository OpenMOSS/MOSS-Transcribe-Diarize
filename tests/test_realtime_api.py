from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from moss_transcribe_diarize.realtime.config import RealtimeConfig
from moss_transcribe_diarize.realtime.store import SessionStore

SCRIPTED_REPLY = "[1.0][S01]你好[2.0]"


class StubEmbedder:
    """确定性向量，够让说话人表真的建起来（WS 用例需要改名/改归属有对象）。"""

    embedding_dim = 2

    def embed(self, audio, sample_rate):
        return np.array([1.0, 0.0], dtype=np.float32)


class ScriptedTranscriber:
    """按窗口返回预置文本的假转写器。"""

    def __init__(self, replies=None):
        self.replies = list(replies or [SCRIPTED_REPLY] * 8)
        self.windows = 0
        self.prompts: list[str] = []  # 让测试能观测 prompt 真的换了

    def transcribe_window(self, audio, *, prompt):
        self.windows += 1
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""


class _Row:
    """``append_committed`` 只需要有 ``to_dict()`` 的对象，这里省掉真的构造 CommittedSegment。"""

    def to_dict(self) -> dict:
        return {
            "id": "seg-1", "start": 1.0, "end": 2.0, "speaker": "S01",
            "speaker_name": "S01", "text": "你好", "speaker_confident": True,
        }


def _app(tmp: Path, *, embedder=None, probe=None, record_audio=True, retranscribe=None,
         static_dir=None, factory=None, **config_kwargs):
    from moss_transcribe_diarize.app.realtime_server import create_realtime_app

    created: list[ScriptedTranscriber] = []

    def build() -> ScriptedTranscriber:
        instance = ScriptedTranscriber()
        created.append(instance)
        return instance

    config = RealtimeConfig(silence_gate=False, **config_kwargs)
    app = create_realtime_app(
        config=config,
        transcriber_factory=factory or build,
        embedder=embedder,
        probe=probe,
        record_audio=record_audio,
        retranscribe=retranscribe,
        static_dir=static_dir,
        runs_dir=tmp / "runs",
    )
    app.state.created_transcribers = created
    return app


class RuntimeRouteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.client = TestClient(_app(Path(self._tmp.name)))

    def test_reports_the_effective_config(self):
        payload = self.client.get("/api/runtime").json()

        self.assertEqual(payload["config"]["window"], 20.0)
        self.assertEqual(payload["config"]["hop"], 5.0)
        self.assertIn("backend", payload)

    def test_reports_speaker_state(self):
        payload = self.client.get("/api/runtime").json()
        self.assertIn("speaker", payload)
        self.assertIn("enabled", payload["speaker"])

    def test_advertises_the_export_formats(self):
        payload = self.client.get("/api/runtime").json()

        self.assertIn("txt", payload["export_formats"])


class SessionsRouteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"
        # 先手工造一个已完成的会话，供读取路由使用
        store = SessionStore(self.runs, "s1", name="周会")
        store.append_committed([_Row()])
        store.write_meta(speakers=[{"id": "S01", "name": "张总", "samples": 2}])
        store.finalize([{"id": "S01", "name": "张总", "samples": 2}])
        self.client = TestClient(_app(Path(self._tmp.name)))

    def test_lists_sessions(self):
        payload = self.client.get("/api/sessions").json()
        self.assertEqual([s["session_id"] for s in payload["sessions"]], ["s1"])

    def test_returns_a_session_with_its_segments(self):
        payload = self.client.get("/api/sessions/s1").json()
        self.assertEqual(payload["session"]["name"], "周会")
        self.assertEqual(payload["segments"][0]["id"], "seg-1")

    def test_unknown_session_is_404_not_500(self):
        self.assertEqual(self.client.get("/api/sessions/nope").status_code, 404)

    def test_a_path_traversal_id_is_rejected_not_served(self):
        response = self.client.get("/api/sessions/..%2F..%2Fetc")
        self.assertIn(response.status_code, (400, 404))

    def test_missing_recording_is_404(self):
        self.assertEqual(self.client.get("/api/sessions/s1/audio").status_code, 404)

    def test_exports_every_advertised_format(self):
        for fmt in ("srt", "json", "ass", "md", "txt"):
            with self.subTest(fmt=fmt):
                response = self.client.get(f"/api/sessions/s1/export?format={fmt}")
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.text.strip())

    def test_export_uses_the_renamed_speaker(self):
        response = self.client.get("/api/sessions/s1/export?format=txt")
        self.assertIn("张总", response.text)

    def test_unknown_export_format_is_400(self):
        self.assertEqual(
            self.client.get("/api/sessions/s1/export?format=docx").status_code, 400
        )

    def test_exporting_an_unknown_session_is_404(self):
        self.assertEqual(
            self.client.get("/api/sessions/nope/export?format=srt").status_code, 404
        )


class WebSocketTest(unittest.TestCase):
    """WS 协议：音频帧、控制指令、每会话驱动。

    音频一律推 `_pcm(9.0)`：默认 ``min_first_window=8.0`` / ``tail=6.0`` 下，9 秒的
    首窗末端为 9.0，定稿裁切线是 3.0，脚本回复的段 (1.0, 2.0) 才落在定稿区里。
    推 1 秒是收不到 committed 事件的——那一段永远在 tail 的临时区里。

    每个开过会话的用例都必须在还留在 ``with`` 里时 ``_stop``：``TestClient`` 退出
    ``with`` 块时会**取消**应用任务，会话收尾就跑不完了。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"
        self.client = TestClient(_app(Path(self._tmp.name), embedder=StubEmbedder()))

    def _pcm(self, seconds: float) -> bytes:
        t = np.arange(int(seconds * 16000), dtype=np.float32) / 16000
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype("<f4").tobytes()

    def _begin(self, ws, **extra) -> str:
        """发 start，返回 session_id。"""
        ws.send_json({"type": "start", **extra})
        return self._drain(ws, want="session")["session_id"]

    def _stop(self, ws, session_id: str) -> None:
        """发 stop 并等会话真的落盘。

        不能只靠"收到一个 speaker 事件"判断：窗口自己发的那些事件还排在队列里，
        先收到的那条多半是旧的。收尾是否真的跑完，看 session.json 的状态最可靠。
        """
        ws.send_json({"type": "stop"})
        self.assertTrue(
            self._wait_until(
                lambda: SessionStore.load_meta(self.runs, session_id).get("status") == "done"
            ),
            "stop 之后会话没有收尾",
        )

    def _wait_until(self, predicate, *, timeout: float = 20.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def _drain(self, ws, *, want: str, where=None, limit: int = 80) -> dict:
        """读到第一条满足条件的指定类型事件为止（别的类型直接跳过）。"""
        for _ in range(limit):
            event = ws.receive_json()
            if event["type"] == want and (where is None or where(event)):
                return event
        self.fail(f"没有收到 {want} 事件")

    def test_first_event_is_the_session_descriptor(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start", "session_name": "测试会议"})
            first = ws.receive_json()
            self.assertEqual(first["type"], "session")
            self.assertTrue(first["session_id"])
            self.assertEqual(first["config"]["window"], 20.0)

    def test_audio_produces_committed_segments(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            session_id = self._begin(ws)
            ws.send_bytes(self._pcm(9.0))
            seen = self._drain(ws, want="committed")
            self.assertTrue(seen["segments"])
            self.assertEqual(seen["segments"][0]["text"], "你好")
            self.assertEqual(seen["segments"][0]["start"], 1.0)
            self._stop(ws, session_id)

    def test_audio_before_start_is_buffered_not_dropped(self):
        """浏览器常常先开麦后发 start，那些音频不该丢。"""
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_bytes(self._pcm(9.0))
            session_id = self._begin(ws)
            seen = self._drain(ws, want="committed")
            self.assertTrue(seen["segments"])
            self._stop(ws, session_id)

    def test_a_misaligned_frame_is_reported_and_the_connection_survives(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            session_id = self._begin(ws)
            ws.send_bytes(b"\x01\x02\x03")           # 不是 4 的倍数
            error = self._drain(ws, want="error")
            self.assertEqual(error["code"], "invalid_audio_frame")
            ws.send_bytes(self._pcm(9.0))            # 连接仍然可用
            self.assertTrue(self._drain(ws, want="committed")["segments"])
            self._stop(ws, session_id)

    def test_stop_closes_the_session_and_persists_it(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            session_id = self._begin(ws)
            ws.send_bytes(self._pcm(9.0))
            self._drain(ws, want="committed")
            self._stop(ws, session_id)

        self.assertTrue(SessionStore.load_committed(self.runs, session_id))

    def test_a_second_start_is_ignored(self):
        """同一条连接上重复 start：不另开会话，也不该报错。"""
        with self.client.websocket_connect("/ws/realtime") as ws:
            session_id = self._begin(ws)
            ws.send_json({"type": "start"})
            ws.send_bytes(self._pcm(9.0))
            self.assertTrue(self._drain(ws, want="committed")["segments"])
            self._stop(ws, session_id)

        self.assertEqual([p.name for p in self.runs.iterdir()], [session_id])

    def test_two_sessions_get_separate_directories(self):
        ids = []
        for _ in range(2):
            with self.client.websocket_connect("/ws/realtime") as ws:
                ids.append(self._begin(ws))
                ws.send_json({"type": "stop"})
        self.assertNotEqual(ids[0], ids[1])
        for session_id in ids:
            self.assertTrue((self.runs / session_id).is_dir())

    def test_two_sessions_run_concurrently_without_interfering(self):
        """两个连接同时在飞：各自独立的会话目录，谁都不该把对方挤崩。"""
        with self.client.websocket_connect("/ws/realtime") as first:
            first_id = self._begin(first)
            with self.client.websocket_connect("/ws/realtime") as second:
                second_id = self._begin(second)
                self.assertNotEqual(first_id, second_id)
                first.send_bytes(self._pcm(9.0))
                second.send_bytes(self._pcm(9.0))

                self.assertTrue(self._drain(first, want="committed")["segments"])
                self.assertTrue(self._drain(second, want="committed")["segments"])
                self._stop(second, second_id)
            self._stop(first, first_id)

        self.assertTrue((self.runs / first_id).is_dir())
        self.assertTrue((self.runs / second_id).is_dir())

    def test_rename_speaker_updates_the_roster(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            session_id = self._begin(ws)
            ws.send_bytes(self._pcm(9.0))
            self._drain(ws, want="committed")
            ws.send_json({"type": "rename_speaker", "speaker_id": "S01", "name": "张总"})
            roster = self._drain(
                ws, want="speaker", where=lambda e: any(i["name"] == "张总" for i in e["speakers"])
            )
            self.assertIn("张总", [item["name"] for item in roster["speakers"]])
            self._stop(ws, session_id)

    def test_renaming_an_unknown_speaker_is_an_error(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            session_id = self._begin(ws)
            ws.send_json({"type": "rename_speaker", "speaker_id": "S99", "name": "谁"})
            self.assertEqual(self._drain(ws, want="error")["code"], "unknown_speaker")
            self._stop(ws, session_id)

    def test_set_hotwords_reaches_the_transcriber(self):
        """模型只认 prompt，没有单独的热词参数——所以热词要拼进 prompt 交给下一次推理。"""
        with self.client.websocket_connect("/ws/realtime") as ws:
            session_id = self._begin(ws)
            ws.send_json({"type": "set_hotwords", "hotwords": ["阿里云", "降噪"]})
            ws.send_bytes(self._pcm(9.0))
            self._drain(ws, want="committed")
            self._stop(ws, session_id)

        prompts = self.client.app.state.created_transcribers[-1].prompts
        self.assertTrue(prompts)
        self.assertIn("阿里云", prompts[-1])
        self.assertIn("降噪", prompts[-1])

    def test_set_prompt_replaces_what_the_next_window_gets(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            session_id = self._begin(ws)
            ws.send_json({"type": "set_prompt", "prompt": "只转写中文"})
            ws.send_bytes(self._pcm(9.0))
            self._drain(ws, want="committed")
            self._stop(ws, session_id)

        prompts = self.client.app.state.created_transcribers[-1].prompts
        self.assertTrue(prompts)
        self.assertEqual(prompts[-1], "只转写中文")

    def test_reassigning_a_segment_emits_it_again_with_the_new_speaker(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            session_id = self._begin(ws)
            ws.send_bytes(self._pcm(9.0))
            segment_id = self._drain(ws, want="committed")["segments"][0]["id"]

            ws.send_json({"type": "reassign_segment", "segment_id": segment_id, "speaker_id": "S01"})

            again = self._drain(
                ws, want="committed", where=lambda e: e["segments"][0]["id"] == segment_id
            )
            self.assertEqual(again["segments"][0]["id"], segment_id)
            self.assertEqual(again["segments"][0]["speaker"], "S01")
            self._stop(ws, session_id)

    def test_reassignment_survives_into_the_stored_transcript(self):
        """改了说话人却不落盘，下次打开就没了——而导出正是读那份文件。"""
        with self.client.websocket_connect("/ws/realtime") as ws:
            session_id = self._begin(ws)
            ws.send_bytes(self._pcm(9.0))
            segment_id = self._drain(ws, want="committed")["segments"][0]["id"]
            ws.send_json({"type": "reassign_segment", "segment_id": segment_id, "speaker_id": "U99"})
            self._drain(
                ws, want="committed", where=lambda e: e["segments"][0]["id"] == segment_id
            )
            self._stop(ws, session_id)

        rows = SessionStore.load_committed(self.runs, session_id)
        self.assertEqual(rows[0]["speaker"], "U99")

    def test_reassigning_an_unknown_segment_is_an_error(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            session_id = self._begin(ws)
            ws.send_json({"type": "reassign_segment", "segment_id": "seg-999", "speaker_id": "S01"})
            self.assertEqual(self._drain(ws, want="error")["code"], "unknown_segment")
            self._stop(ws, session_id)

    def test_control_before_start_is_an_error_not_a_crash(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "reassign_segment", "segment_id": "seg-1", "speaker_id": "S01"})
            self.assertEqual(self._drain(ws, want="error")["code"], "no_session")

    def test_unknown_control_message_is_an_error_not_a_crash(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            session_id = self._begin(ws)
            ws.send_json({"type": "wat"})
            self.assertEqual(self._drain(ws, want="error")["code"], "unknown_command")
            self._stop(ws, session_id)

    def test_broken_json_is_reported_not_fatal(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            session_id = self._begin(ws)
            ws.send_text("{not json")
            self.assertEqual(self._drain(ws, want="error")["code"], "invalid_json")
            self._stop(ws, session_id)


class DisconnectTest(unittest.TestCase):
    """断开连接也要收尾。

    ``TestClient`` 在退出 ``with`` 块时会**取消**应用任务，所以这条路径在它下面永远
    走不到最后——这里用裸 ASGI 驱动，把 ``websocket.disconnect`` 真的喂进去。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"
        self.app = _app(Path(self._tmp.name), embedder=StubEmbedder())

    def _pcm(self, seconds: float) -> bytes:
        t = np.arange(int(seconds * 16000), dtype=np.float32) / 16000
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype("<f4").tobytes()

    def _drive(self, messages: list[dict]) -> list[dict]:
        import asyncio

        app = self.app
        sent: list[dict] = []

        async def main() -> None:
            incoming: asyncio.Queue = asyncio.Queue()
            for message in messages:
                incoming.put_nowait(message)

            async def receive() -> dict:
                return await incoming.get()

            async def send(message: dict) -> None:
                sent.append(message)

            scope = {
                "type": "websocket",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "scheme": "ws",
                "path": "/ws/realtime",
                "raw_path": b"/ws/realtime",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"host", b"testserver")],
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
                "subprotocols": [],
            }
            await asyncio.wait_for(app(scope, receive, send), timeout=60)

        asyncio.run(main())
        return sent

    def test_a_disconnect_finalizes_the_session(self):
        messages = [
            {"type": "websocket.connect"},
            {"type": "websocket.receive", "text": '{"type":"start","session_name":"断开测试"}'},
            {"type": "websocket.receive", "bytes": self._pcm(9.0)},
            {"type": "websocket.disconnect", "code": 1000, "reason": None},
        ]

        events = self._drive(messages)

        payloads = [
            json.loads(message["text"])
            for message in events
            if message["type"] == "websocket.send" and "text" in message
        ]
        kinds = [payload["type"] for payload in payloads]
        self.assertIn("session", kinds)
        self.assertIn("committed", kinds)
        session_id = payloads[0]["session_id"]
        meta = SessionStore.load_meta(self.runs, session_id)
        self.assertEqual(meta["status"], "done")
        self.assertEqual(meta["name"], "断开测试")
        self.assertTrue(SessionStore.load_committed(self.runs, session_id))


class TrimPendingTest(unittest.TestCase):
    """``start`` 之前代管的音频有上限：一个从不发 ``start`` 的客户端不该能把内存堆满。"""

    def _trim(self, sizes, limit):
        from moss_transcribe_diarize.app.realtime_server import _trim_pending

        pending = [np.zeros(size, dtype=np.float32) for size in sizes]
        remaining = _trim_pending(pending, limit)
        return [frame.size for frame in pending], remaining

    def test_keeps_everything_below_the_limit(self):
        frames, remaining = self._trim([100, 100], 1000)

        self.assertEqual(frames, [100, 100])
        self.assertEqual(remaining, 200)

    def test_drops_the_oldest_frames_first(self):
        frames, remaining = self._trim([100, 100, 100], 250)

        self.assertEqual(frames, [100, 100])
        self.assertEqual(remaining, 200)

    def test_keeps_the_newest_frame_even_when_it_alone_exceeds_the_limit(self):
        """单帧就超上限时也不能把待处理列表清空——那就真的一帧都不剩了。"""
        frames, remaining = self._trim([1000], 10)

        self.assertEqual(frames, [1000])
        self.assertEqual(remaining, 1000)


class BackendProbeTest(unittest.TestCase):
    """忘起 vLLM 是一次常见失误；服务要在开始之前就能把它说清楚。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)

    def _runtime(self, probe):
        return TestClient(_app(Path(self._tmp.name), probe=probe)).get("/api/runtime").json()

    def test_reports_an_unreachable_endpoint(self):
        payload = self._runtime(lambda: {"reachable": False, "detail": "connection refused"})

        self.assertFalse(payload["backend"]["reachable"])
        self.assertIn("connection refused", payload["backend"]["detail"])

    def test_reports_a_reachable_endpoint(self):
        payload = self._runtime(lambda: {"reachable": True, "detail": "ok"})

        self.assertTrue(payload["backend"]["reachable"])

    def test_a_probe_that_blows_up_does_not_take_the_service_down(self):
        def broken():
            raise RuntimeError("probe exploded")

        payload = self._runtime(broken)

        self.assertFalse(payload["backend"]["reachable"])
        self.assertIn("probe exploded", payload["backend"]["detail"])

    def test_without_a_probe_there_is_no_reachability_claim(self):
        payload = self._runtime(None)

        self.assertNotIn("reachable", payload["backend"])
        self.assertIn("factory", payload["backend"])


class CheckEndpointTest(unittest.TestCase):
    """``check_endpoint`` 只报告结果，从不抛。"""

    def test_an_unreachable_port_is_reported_not_raised(self):
        from moss_transcribe_diarize.app.openai_audio_client import check_endpoint

        result = check_endpoint("http://127.0.0.1:9", timeout=1.0)

        self.assertFalse(result["reachable"])
        self.assertTrue(result["detail"])

    def test_it_probes_the_models_route_not_the_transcriptions_route(self):
        import urllib.request

        from moss_transcribe_diarize.app.openai_audio_client import check_endpoint

        seen: list[str] = []
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
        self.addCleanup(setattr, urllib.request, "urlopen", original)

        result = check_endpoint("http://host:8000/v1/audio/transcriptions")

        self.assertTrue(result["reachable"])
        self.assertEqual(seen, ["http://host:8000/v1/models"])


class NoRecordTest(unittest.TestCase):
    """``--no-record``：只留转写文本，不落盘会议录音（那是敏感数据）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"
        self.client = TestClient(
            _app(Path(self._tmp.name), embedder=StubEmbedder(), record_audio=False)
        )

    def test_no_recording_is_written_but_the_transcript_is(self):
        t = np.arange(int(9.0 * 16000), dtype=np.float32) / 16000
        pcm = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("<f4").tobytes()

        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start"})
            session_id = ws.receive_json()["session_id"]
            ws.send_bytes(pcm)
            for _ in range(40):
                if ws.receive_json()["type"] == "committed":
                    break
            ws.send_json({"type": "stop"})

        self.assertFalse((self.runs / session_id / "audio.wav").exists())
        self.assertTrue((self.runs / session_id / "transcript.jsonl").exists())
        meta = SessionStore.load_meta(self.runs, session_id)
        self.assertFalse(meta["record_audio"])


class RetranscribeTest(unittest.TestCase):
    """spec §4.8 / §5.2 的"重跑本次会话"：把整段录音交给当前后端跑一次文件模式。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"
        store = SessionStore(self.runs, "s1", name="周会")
        store.append_committed([_Row()])
        store.append_audio(np.zeros(16000, dtype=np.float32))
        store.finalize([])

    def _client(self, retranscribe=None):
        return TestClient(_app(Path(self._tmp.name), retranscribe=retranscribe))

    def test_runs_the_whole_recording_through_the_injected_backend(self):
        seen = []

        def fake(path, prompt):
            seen.append((Path(path).name, prompt))
            return "[0.5][S01]重跑的结果[3.0]"

        response = self._client(fake).post("/api/sessions/s1/retranscribe")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "[0.5][S01]重跑的结果[3.0]")
        self.assertEqual(seen, [("audio.wav", "")])

    def test_a_failing_backend_is_reported_with_its_reason(self):
        def broken(path, prompt):
            raise RuntimeError("vLLM 没起来")

        response = self._client(broken).post("/api/sessions/s1/retranscribe")

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["code"], "retranscribe_failed")
        self.assertIn("vLLM 没起来", response.json()["detail"])

    def test_without_a_backend_it_says_so_instead_of_crashing(self):
        response = self._client(None).post("/api/sessions/s1/retranscribe")

        self.assertEqual(response.status_code, 501)
        self.assertEqual(response.json()["code"], "retranscribe_unavailable")

    def test_the_sessions_own_prompt_is_reused(self):
        """重跑必须用实时那次用过的 prompt，否则重跑出来的内容与当场那次不是一回事。"""
        store = SessionStore(self.runs, "s1", name="周会")
        store.write_meta(prompt="只转写中文")
        store.finalize([])          # 重开会话会把状态写回 recording，这里再收一次尾

        seen = []
        self._client(lambda path, prompt: seen.append(prompt) or "x").post(
            "/api/sessions/s1/retranscribe"
        )

        self.assertEqual(seen, ["只转写中文"])

    def test_unknown_session_is_404(self):
        self.assertEqual(
            self._client(lambda path, prompt: "x").post("/api/sessions/nope/retranscribe").status_code,
            404,
        )

    def test_a_session_without_a_recording_is_404(self):
        SessionStore(self.runs, "s2", name="没录音").finalize([])

        response = self._client(lambda path, prompt: "x").post("/api/sessions/s2/retranscribe")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "audio_missing")


class StaticAssetTest(unittest.TestCase):
    """前端要的 .js / .css / .json 得有出口，且出口不能变成任意文件读取。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.assets = root / "assets"
        self.assets.mkdir(parents=True)
        (self.assets / "realtime.html").write_text("<!doctype html><title>rt</title>", encoding="utf-8")
        (self.assets / "realtime.js").write_text("export const x = 1;", encoding="utf-8")
        (self.assets / "audio-worklet.js").write_text("registerProcessor('x', class {});", encoding="utf-8")
        (self.assets / "realtime.css").write_text("body { margin: 0 }", encoding="utf-8")
        (self.assets / "locales").mkdir()
        (self.assets / "locales" / "zh-CN.json").write_text('{"a": "b"}', encoding="utf-8")
        (root / "secret.txt").write_text("nope", encoding="utf-8")
        self.client = TestClient(_app(root, static_dir=self.assets))

    def test_serves_the_frontend_files_with_useful_content_types(self):
        for name, media in (
            ("realtime.js", "text/javascript"),
            ("audio-worklet.js", "text/javascript"),
            ("realtime.css", "text/css"),
            ("locales/zh-CN.json", "application/json"),
        ):
            with self.subTest(name=name):
                response = self.client.get(f"/assets/{name}")
                self.assertEqual(response.status_code, 200)
                self.assertIn(media, response.headers["content-type"])

    def test_serves_the_page_at_the_root(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("<title>rt</title>", response.text)

    def test_a_missing_asset_is_404(self):
        self.assertEqual(self.client.get("/assets/nope.js").status_code, 404)

    def test_assets_are_not_cached(self):
        """改了代码却因为浏览器缓存没生效，是一次很难查的玄学——本地服务不该缓存。"""
        response = self.client.get("/assets/realtime.js")

        self.assertEqual(response.headers.get("cache-control"), "no-store")

    def test_a_directory_is_404_not_a_listing(self):
        self.assertEqual(self.client.get("/assets/locales").status_code, 404)

    def test_a_traversal_attempt_cannot_read_outside_the_assets_dir(self):
        for attempt in ("..%2F..%2Fsecret.txt", "..%2Fsecret.txt", "%2e%2e%2f%2e%2e%2fsecret.txt"):
            with self.subTest(attempt=attempt):
                response = self.client.get(f"/assets/{attempt}")
                self.assertIn(response.status_code, (400, 404))
                self.assertNotIn("nope", response.text)

class BackendFailureTest(unittest.TestCase):
    """后端起不来（模型路径写错、显存不够）必须是可观测的失败。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def _client(self, factory):
        return TestClient(_app(Path(self._tmp.name), embedder=StubEmbedder(), factory=factory))

    def test_a_failing_factory_is_reported_not_swallowed(self):
        def boom():
            raise RuntimeError("model weights not found at /nope")

        with self._client(boom).websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start", "session_name": "现场会"})
            first = ws.receive_json()

        self.assertEqual(first["type"], "error")
        self.assertEqual(first["code"], "backend_unavailable")
        self.assertIn("model weights not found", first["detail"])

    def test_a_failing_factory_leaves_no_ghost_session_behind(self):
        """否则历史里会多出一条永远停在 recording、既没录音也没转写的会话。"""
        def boom():
            raise RuntimeError("out of memory")

        with self._client(boom).websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start", "session_name": "现场会"})
            ws.receive_json()

        self.assertEqual(
            [p.name for p in self.runs.iterdir()] if self.runs.is_dir() else [], []
        )


class ControlCommandsAreSerializedTest(unittest.TestCase):
    """会话有三个写者：驱动、收尾、控制指令。

    `reassign_segment` 的 `rewrite_committed` 是**整体重写** `transcript.jsonl`，撞上正在
    追加的窗口就会把刚定稿的那段永久抹掉——不报任何错。这条测试钉住那个不变量：窗口在飞
    的时候，控制指令不许改动会话。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def test_a_rename_never_lands_while_a_window_is_in_flight(self):
        import threading
        from unittest import mock

        from moss_transcribe_diarize.realtime.session import RealtimeSession

        gate = threading.Event()
        entered_second = threading.Event()

        class BlockingTranscriber:
            def __init__(self):
                self.calls = 0

            def transcribe_window(self, audio, *, prompt):
                self.calls += 1
                if self.calls == 1:
                    return SCRIPTED_REPLY
                entered_second.set()
                gate.wait(20)
                return "[1.0][S01]第二段[2.0]"

        window_open = threading.Event()
        offenders: list = []
        real_run_pending = RealtimeSession.run_pending
        real_rename = RealtimeSession.rename_speaker

        async def watched_run_pending(self):
            window_open.set()
            try:
                return await real_run_pending(self)
            finally:
                window_open.clear()

        def watched_rename(self, speaker_id, name):
            if window_open.is_set():
                offenders.append((speaker_id, name))
            return real_rename(self, speaker_id, name)

        blocker = BlockingTranscriber()
        client = TestClient(
            _app(Path(self._tmp.name), embedder=StubEmbedder(), factory=lambda: blocker)
        )

        with mock.patch.object(RealtimeSession, "run_pending", watched_run_pending), \
                mock.patch.object(RealtimeSession, "rename_speaker", watched_rename):
            with client.websocket_connect("/ws/realtime") as ws:
                ws.send_json({"type": "start"})
                session_id = ws.receive_json()["session_id"]
                ws.send_bytes(_pcm_bytes(9.0))
                self._drain(ws, want="committed")

                ws.send_bytes(_pcm_bytes(9.0))
                self.assertTrue(entered_second.wait(20), "第二个窗口没跑起来")

                ws.send_json({"type": "rename_speaker", "speaker_id": "S01", "name": "张总"})
                time.sleep(0.4)          # 不阻塞的话，这条指令此刻已经改完会话了
                gate.set()
                self._drain(ws, want="speaker",
                            where=lambda e: any(i["name"] == "张总" for i in e["speakers"]))
                ws.send_json({"type": "stop"})
                # 收尾要在还留在 with 里时等完：TestClient 退出 with 时会取消应用任务。
                self.assertTrue(self._wait_until(
                    lambda: SessionStore.load_meta(self.runs, session_id).get("status") == "done"
                ))

        self.assertEqual(offenders, [], "控制指令在窗口在飞的时候改了会话")

    def _drain(self, ws, *, want, where=None, limit=80):
        for _ in range(limit):
            event = ws.receive_json()
            if event["type"] == want and (where is None or where(event)):
                return event
        self.fail(f"没有收到 {want} 事件")

    def _wait_until(self, predicate, *, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False


class RenameIsPersistedTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"
        self.client = TestClient(_app(Path(self._tmp.name), embedder=StubEmbedder()))

    def test_an_export_taken_mid_session_uses_the_new_name(self):
        """导出读的是 session.json 里的说话人表，而它平时只在 finalize 时才写。"""
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start"})
            session_id = ws.receive_json()["session_id"]
            ws.send_bytes(_pcm_bytes(9.0))
            self._drain(ws, want="committed")
            ws.send_json({"type": "rename_speaker", "speaker_id": "S01", "name": "张总"})
            self._drain(ws, want="speaker",
                        where=lambda e: any(i["name"] == "张总" for i in e["speakers"]))

            # 会话还在录，这里就导出
            exported = self.client.get(f"/api/sessions/{session_id}/export?format=txt").text

            ws.send_json({"type": "stop"})

        self.assertIn("张总", exported)

    def _drain(self, ws, *, want, where=None, limit=80):
        for _ in range(limit):
            event = ws.receive_json()
            if event["type"] == want and (where is None or where(event)):
                return event
        self.fail(f"没有收到 {want} 事件")


class HotwordsReplaceTest(unittest.TestCase):
    """``set_hotwords`` 是**替换**语义，不是追加。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"
        self.client = TestClient(_app(Path(self._tmp.name), embedder=StubEmbedder()))

    def _prompt_after(self, commands):
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start"})
            session_id = ws.receive_json()["session_id"]
            for command in commands:
                ws.send_json(command)
            ws.send_bytes(_pcm_bytes(9.0))
            self._drain(ws, want="committed")
            ws.send_json({"type": "stop"})
        return self.client.app.state.created_transcribers[-1].prompts[-1]

    def test_the_second_set_replaces_the_first(self):
        prompt = self._prompt_after([
            {"type": "set_hotwords", "hotwords": ["阿里云"]},
            {"type": "set_hotwords", "hotwords": ["降噪"]},
        ])

        self.assertIn("降噪", prompt)
        self.assertNotIn("阿里云", prompt)

    def test_empty_hotwords_clear_the_previous_ones(self):
        prompt = self._prompt_after([
            {"type": "set_hotwords", "hotwords": ["阿里云"]},
            {"type": "set_hotwords", "hotwords": []},
        ])

        self.assertNotIn("阿里云", prompt)

    def test_set_prompt_keeps_the_hotwords_it_was_given_with(self):
        prompt = self._prompt_after([
            {"type": "set_prompt", "prompt": "只转写中文", "hotwords": ["阿里云"]},
        ])

        self.assertIn("只转写中文", prompt)
        self.assertIn("阿里云", prompt)

    def _drain(self, ws, *, want, where=None, limit=80):
        for _ in range(limit):
            event = ws.receive_json()
            if event["type"] == want and (where is None or where(event)):
                return event
        self.fail(f"没有收到 {want} 事件")


class NonFiniteFrameTest(unittest.TestCase):
    """把 int16 采样当 float32 解，长度刚好是 4 的倍数，但内容是 NaN。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"
        self.client = TestClient(_app(Path(self._tmp.name), embedder=StubEmbedder()))

    def test_a_frame_full_of_nan_is_rejected_and_the_connection_survives(self):
        int16 = np.arange(-1600, 1600, dtype=np.int16)
        misread = np.frombuffer(int16.tobytes(), dtype="<f4")
        self.assertEqual(int16.nbytes % 4, 0)
        self.assertFalse(np.isfinite(misread).all())

        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start"})
            session_id = ws.receive_json()["session_id"]

            ws.send_bytes(misread.tobytes())
            error = self._drain(ws, want="error")
            self.assertEqual(error["code"], "invalid_audio_frame")

            ws.send_bytes(_pcm_bytes(9.0))          # 连接仍然可用
            self.assertTrue(self._drain(ws, want="committed")["segments"])
            ws.send_json({"type": "stop"})

    def _drain(self, ws, *, want, where=None, limit=80):
        for _ in range(limit):
            event = ws.receive_json()
            if event["type"] == want and (where is None or where(event)):
                return event
        self.fail(f"没有收到 {want} 事件")


class RetranscribeWhileRecordingTest(unittest.TestCase):
    """录音还在写的时候重跑，读到的是一份 WAV 头与内容对不上的半成品。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def test_a_session_that_is_still_recording_is_409(self):
        store = SessionStore(self.runs, "live", name="进行中")
        store.append_audio(np.zeros(16000, dtype=np.float32))

        client = TestClient(_app(Path(self._tmp.name), retranscribe=lambda p, q: "不该被调用"))
        response = client.post("/api/sessions/live/retranscribe")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "session_busy")

    def test_a_finished_session_still_re_runs(self):
        store = SessionStore(self.runs, "done", name="已结束")
        store.append_audio(np.zeros(16000, dtype=np.float32))
        store.finalize([])

        client = TestClient(_app(Path(self._tmp.name), retranscribe=lambda p, q: "重跑结果"))
        response = client.post("/api/sessions/done/retranscribe")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "重跑结果")


def _pcm_bytes(seconds: float) -> bytes:
    t = np.arange(int(seconds * 16000), dtype=np.float32) / 16000
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype("<f4").tobytes()


if __name__ == "__main__":
    unittest.main()

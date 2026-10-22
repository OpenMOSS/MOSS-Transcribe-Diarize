from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
import weakref
from pathlib import Path

import numpy as np

from moss_transcribe_diarize.realtime.config import RealtimeConfig
from moss_transcribe_diarize.realtime.session import RealtimeSession, segment_audio
from moss_transcribe_diarize.realtime.stitch import Segment
from moss_transcribe_diarize.realtime.store import SessionStore

SILENCE = np.zeros(0, dtype=np.float32)  # 占位，实际音频由 _speech/_silence 生成


def _speech(seconds: float, *, amplitude: float = 0.3, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(round(seconds * sample_rate)), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _silence(seconds: float, *, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(round(seconds * sample_rate)), dtype=np.float32)


class ScriptedTranscriber:
    """按调用次序返回预设文本，并记录每次收到的窗口时长。"""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.window_seconds: list[float] = []

    def transcribe_window(self, audio, *, prompt):
        self.window_seconds.append(len(audio) / 16000.0)
        if not self.replies:
            return ""
        return self.replies.pop(0)


class FailingTranscriber:
    def __init__(self, message: str = "boom"):
        self.message = message
        self.calls = 0

    def transcribe_window(self, audio, *, prompt):
        self.calls += 1
        raise RuntimeError(self.message)


class FlakyEmbedder:
    """第一次嵌入抛异常，之后正常。用于验证定稿失败后的回滚。"""

    embedding_dim = 2

    def __init__(self):
        self.calls = 0

    def embed(self, audio, sample_rate):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("声纹模型挂了")
        return np.array([1.0, 0.0], dtype=np.float32)


class SpanRecordingSession(RealtimeSession):
    """记录每一次**真正执行**的推理窗口区间（被门控跳过的不算）。

    窗口的起止是 ``run_pending`` 内部算出来的，事件里不带，所以"哪些音频被覆盖过"
    只能从 ``_run_window`` 的入参观测。测试里覆盖这一个私有方法，是为了拿到真实发生过
    的区间，而不是按公式重新推算一遍（那样会把待验证的公式写进断言里）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spans: list[tuple[float, float]] = []

    async def _run_window(self, start, end, audio):
        self.spans.append((start, end))
        return await super()._run_window(start, end, audio)


def _config(**kwargs) -> RealtimeConfig:
    params = {
        "window": 20.0,
        "hop": 5.0,
        "tail": 6.0,
        "min_first_window": 8.0,
        "silence_gate": False,
    }
    params.update(kwargs)
    return RealtimeConfig(**params)


class RealtimeSessionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def _session(self, transcriber, config=None, **store_kwargs) -> RealtimeSession:
        return RealtimeSession(
            config or _config(),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1", **store_kwargs),
        )

    def _run(self, coro):
        return asyncio.run(coro)

    def test_does_not_run_before_min_first_window(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(5.0))

        events = self._run(session.run_pending())

        self.assertEqual(events, [])
        self.assertEqual(transcriber.window_seconds, [])

    def test_empty_buffer_returns_no_events(self):
        """缓冲里一帧音频都没有时，绝不能把空音频送进处理器。"""
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"])
        session = self._session(transcriber)

        events = self._run(session.run_pending())

        self.assertEqual(events, [])
        self.assertEqual(transcriber.window_seconds, [])

    def test_empty_push_is_ignored(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"])
        session = self._session(transcriber)
        session.push_audio(np.zeros(0, dtype=np.float32))

        self.assertEqual(self._run(session.run_pending()), [])
        self.assertEqual(transcriber.window_seconds, [])

    def test_first_window_runs_at_min_first_window(self):
        transcriber = ScriptedTranscriber(["[1][S01]开场[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        self.assertAlmostEqual(transcriber.window_seconds[0], 8.0, places=2)
        self.assertIn("committed", [event["type"] for event in events])

    def test_committed_segments_carry_ids_and_absolute_times(self):
        transcriber = ScriptedTranscriber(["[1][S01]你好[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        committed = next(event for event in events if event["type"] == "committed")
        self.assertEqual(committed["segments"][0]["id"], "seg-1")
        self.assertAlmostEqual(committed["segments"][0]["start"], 1.0, places=2)
        self.assertAlmostEqual(committed["segments"][0]["end"], 2.0, places=2)
        self.assertEqual(committed["segments"][0]["text"], "你好")

    def test_provisional_event_is_emitted_even_when_empty(self):
        transcriber = ScriptedTranscriber(["[1][S01]你好[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        provisional = next(event for event in events if event["type"] == "provisional")
        self.assertEqual(provisional["segments"], [])

    def test_late_segment_stays_provisional_then_commits_later(self):
        # 第一次窗口 [0,20]，段落落在 18-19 秒，在 tail=6 之内，所以只能是临时段。
        # 第二次窗口是 [10,30]，同一段内容此时落在窗口内的 8-9 秒处，换算回绝对
        # 时间仍是 18-19 秒，而 cutoff 是 30-6=24，于是被定稿。
        transcriber = ScriptedTranscriber(["[18][S01]结尾[19]", "[8][S01]结尾[9]"])
        session = self._session(transcriber)
        session.push_audio(_speech(20.0))
        events = self._run(session.run_pending())
        self.assertEqual([e for e in events if e["type"] == "committed"], [])

        session.push_audio(_speech(10.0))
        events = self._run(session.run_pending())

        committed = next(event for event in events if event["type"] == "committed")
        self.assertEqual([seg["text"] for seg in committed["segments"]], ["结尾"])
        self.assertAlmostEqual(committed["segments"][0]["start"], 18.0, places=2)
        self.assertAlmostEqual(committed["segments"][0]["end"], 19.0, places=2)

    def test_hop_throttles_repeated_runs(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 5)
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))
        self._run(session.run_pending())

        session.push_audio(_speech(2.0))
        events = self._run(session.run_pending())

        self.assertEqual(events, [])
        self.assertEqual(len(transcriber.window_seconds), 1)

    def test_window_covers_at_most_window_seconds(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 5)
        session = self._session(transcriber)
        session.push_audio(_speech(60.0))

        self._run(session.run_pending())
        session.push_audio(_speech(10.0))
        self._run(session.run_pending())

        self.assertAlmostEqual(transcriber.window_seconds[1], 20.0, places=2)

    def test_records_audio_to_store(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))
        self._run(session.run_pending())

        events = self._run(session.close())
        del events

        audio_path = self.runs / "s1" / "audio.wav"
        self.assertTrue(audio_path.exists())
        import soundfile as sf

        written, _ = sf.read(str(audio_path), dtype="float32")
        self.assertEqual(written.shape[0], 8 * 16000)

    def test_close_flushes_provisional_and_finalizes(self):
        transcriber = ScriptedTranscriber(["[18][S01]结尾[19]", "[18][S01]结尾[19]"])
        session = self._session(transcriber)
        session.push_audio(_speech(20.0))
        self._run(session.run_pending())

        events = self._run(session.close())

        committed = next(event for event in events if event["type"] == "committed")
        self.assertEqual([seg["text"] for seg in committed["segments"]], ["结尾"])
        self.assertTrue(session.closed)

    def test_close_is_idempotent(self):
        transcriber = ScriptedTranscriber([])
        session = self._session(transcriber)

        self._run(session.close())

        self.assertEqual(self._run(session.close()), [])

    def test_push_after_close_is_ignored(self):
        transcriber = ScriptedTranscriber([])
        session = self._session(transcriber)
        self._run(session.close())

        session.push_audio(_speech(8.0))

        self.assertEqual(session.committed, [])

    def test_status_event_reports_transcript_lag(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 5)
        session = self._session(transcriber)
        session.push_audio(_speech(30.0))

        events = self._run(session.run_pending())

        status = next(event for event in events if event["type"] == "status")
        # 首次运行的窗口是 [0, 30]：Ruling 15 的读法 B 规定首窗左边界恒为 0，不能套用
        # 后续窗口的 max(0, total - window)。段落局部 1-2 秒即绝对 1-2 秒，低于
        # cutoff = 30 - 6 = 24，故被定稿、水位线推到 2.0。字幕落后 = 缓冲 30 - 定稿 2 = 28。
        self.assertAlmostEqual(status["lag_sec"], 28.0, places=1)
        self.assertAlmostEqual(status["buffered_sec"], 30.0, places=1)
        self.assertIn("rtf", status)

    def test_single_failure_reports_error_and_keeps_going(self):
        transcriber = FailingTranscriber()
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        error = next(event for event in events if event["type"] == "error")
        self.assertEqual(error["code"], "transcribe_failed")
        self.assertIn("boom", error["detail"])

    def test_repeated_failures_mark_degraded(self):
        transcriber = FailingTranscriber()
        session = self._session(transcriber)
        for _ in range(3):
            session.push_audio(_speech(10.0))
            events = self._run(session.run_pending())

        status = next(event for event in events if event["type"] == "status")
        self.assertTrue(status["degraded"])

    def test_a_success_resets_the_failure_counter(self):
        class FlakyTranscriber:
            def __init__(self):
                self.calls = 0

            def transcribe_window(self, audio, *, prompt):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("第一次失败")
                return "[1][S01]恢复[2]"

        session = self._session(FlakyTranscriber())
        session.push_audio(_speech(10.0))
        self._run(session.run_pending())

        session.push_audio(_speech(10.0))
        events = self._run(session.run_pending())

        status = next(event for event in events if event["type"] == "status")
        self.assertFalse(status["degraded"])

    def test_speaker_event_lists_the_roster(self):
        transcriber = ScriptedTranscriber(["[1][S01]你好[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        speaker = next(event for event in events if event["type"] == "speaker")
        self.assertEqual(speaker["speakers"], [])

    def test_audio_of_uses_window_offsets_not_the_buffer(self):
        """段落音频按窗口内偏移切取，避免环形缓冲回绕后取错位置。"""
        class RecordingSpeakerEmbedder:
            embedding_dim = 2
            seen_sizes: list[int] = []

            def embed(self, audio, sample_rate):
                RecordingSpeakerEmbedder.seen_sizes.append(int(np.asarray(audio).size))
                return np.array([1.0, 0.0], dtype=np.float32)

        RecordingSpeakerEmbedder.seen_sizes = []
        transcriber = ScriptedTranscriber(["[1][S01]你好[2]"])
        session = RealtimeSession(
            _config(),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1"),
            embedder=RecordingSpeakerEmbedder(),
        )
        session.push_audio(_speech(8.0))

        self._run(session.run_pending())

        self.assertEqual(RecordingSpeakerEmbedder.seen_sizes, [16000])

    def test_hop_measures_from_the_last_run_not_the_last_poll(self):
        """被跳过的轮询不得推进 _last_run_sec。

        跳过一次之后必须还能按"距上次真正跑过的窗口"重新触发；若在未运行的分支里
        顺手把 last_run 推到当前缓冲末尾，节流会把下一次本该跑的窗口也吃掉。
        """
        transcriber = ScriptedTranscriber(["[1][S01]a[2]", "[3][S01]b[4]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))
        self._run(session.run_pending())  # 窗口 [0, 8]，last_run = 8

        session.push_audio(_speech(2.0))  # 总长 10，距 8 只有 2 秒 → 跳过
        self.assertEqual(self._run(session.run_pending()), [])

        session.push_audio(_speech(4.0))  # 总长 14，距 8 已 6 秒 ≥ hop → 必须跑
        events = self._run(session.run_pending())

        # 第三窗是 [0, 14]，段落 3-4 秒越过水位线 2.0 且低于 cutoff 8，故被定稿。
        self.assertIn("committed", [event["type"] for event in events])
        self.assertEqual(len(transcriber.window_seconds), 2)

    def test_segment_audio_comes_from_the_window_copy_after_the_buffer_evicts_it(self):
        """推理期间缓冲会继续写入并淘汰窗口音频，段落音频必须取自窗口副本。

        真实场景里一次窗口推理要花几秒到几十秒，网络层在此期间仍在推流。这里让转写器
        在 transcribe_window 里推入超过容量的音频，把窗口覆盖的那段从环形缓冲中淘汰，
        于是"改用缓冲重新切取"会取不到音频（slice 返回 None），声纹嵌入根本不会被调用。
        """
        embedder_sizes: list[int] = []

        class RecordingSpeakerEmbedder:
            embedding_dim = 2

            def embed(self, audio, sample_rate):
                embedder_sizes.append(int(np.asarray(audio).size))
                return np.array([1.0, 0.0], dtype=np.float32)

        class PushingTranscriber:
            """转写期间继续推流的转写器，模拟推理期间网络层仍在写入。"""

            def __init__(self):
                self.session_ref = None
                self.window_seconds: list[float] = []

            def transcribe_window(self, audio, *, prompt):
                self.window_seconds.append(len(audio) / 16000.0)
                session = self.session_ref() if self.session_ref is not None else None
                if session is not None:
                    session.push_audio(_speech(40.0))
                return "[1][S01]你好[2]"

        transcriber = PushingTranscriber()
        session = RealtimeSession(
            _config(buffer_capacity=30.0),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1"),
            embedder=RecordingSpeakerEmbedder(),
        )
        # 弱引用：转写器若强引用会话就形成环，WAV 句柄会活过 TemporaryDirectory.cleanup。
        transcriber.session_ref = weakref.ref(session)
        session.push_audio(_speech(8.0))

        self._run(session.run_pending())

        self.assertEqual(len(transcriber.window_seconds), 1)
        self.assertEqual(embedder_sizes, [16000])

    def test_recovery_after_degrading_clears_the_counter(self):
        """连续失败到 degraded 之后，一次成功必须把 degraded 清掉。

        只做"一次失败 + 一次成功"是测不出计数未复位的：那时计数只有 1，远低于阈值，
        两种实现都给 False。必须先真的跨过阈值。
        """
        class EventuallyWorkingTranscriber:
            def __init__(self):
                self.calls = 0

            def transcribe_window(self, audio, *, prompt):
                self.calls += 1
                if self.calls <= 3:
                    raise RuntimeError("连续失败")
                return "[1][S01]恢复[2]"

        session = self._session(EventuallyWorkingTranscriber())
        for _ in range(3):
            session.push_audio(_speech(10.0))
            events = self._run(session.run_pending())

        status = next(event for event in events if event["type"] == "status")
        self.assertTrue(status["degraded"])

        session.push_audio(_speech(10.0))
        events = self._run(session.run_pending())

        status = next(event for event in events if event["type"] == "status")
        self.assertFalse(status["degraded"])

    def test_provisional_event_replaces_rather_than_accumulates(self):
        """provisional 是整体替换：第二窗没有临时段时事件必须是空表。"""
        transcriber = ScriptedTranscriber(["[18][S01]结尾[19]", "[8][S01]结尾[9]"])
        session = self._session(transcriber)
        session.push_audio(_speech(20.0))
        first = self._run(session.run_pending())

        first_provisional = next(event for event in first if event["type"] == "provisional")
        self.assertEqual([seg["text"] for seg in first_provisional["segments"]], ["结尾"])

        session.push_audio(_speech(10.0))
        second = self._run(session.run_pending())

        second_provisional = next(event for event in second if event["type"] == "provisional")
        self.assertEqual(second_provisional["segments"], [])

    def test_close_uses_a_fresh_window_result_when_promoting(self):
        """close() 必须重跑一次推理再定稿，不能直接提升上一窗的临时快照。

        两个窗口给出不同文本，定稿内容因此能区分"新推理结果"与"旧快照"。
        """
        transcriber = ScriptedTranscriber(["[18][S01]旧[19]", "[18][S01]新[19]"])
        session = self._session(transcriber)
        session.push_audio(_speech(20.0))
        self._run(session.run_pending())

        events = self._run(session.close())

        committed = next(event for event in events if event["type"] == "committed")
        self.assertEqual([seg["text"] for seg in committed["segments"]], ["新"])
        self.assertEqual(len(transcriber.window_seconds), 2)

    def test_close_runs_the_blocking_teardown_off_the_event_loop_thread(self):
        """close() 的收尾（声纹嵌入、关 WAV、写元信息）必须在工作线程里跑。

        事件内容在两种实现下完全一样，所以"跑在哪个线程上"是这条路径唯一可观测的差
        异，只能记录线程标识。两个窗口都只产出临时段，逐窗路径不会调用嵌入器，于是嵌
        入器唯一的一次调用必然来自 close() 的 flush；若收尾退回事件循环线程，嵌入器
        看到的 ident 就等于循环线程的 ident。
        """
        class ThreadRecordingEmbedder:
            embedding_dim = 2
            idents: list[int] = []

            def embed(self, audio, sample_rate):
                ThreadRecordingEmbedder.idents.append(threading.get_ident())
                return np.array([1.0, 0.0], dtype=np.float32)

        ThreadRecordingEmbedder.idents = []
        transcriber = ScriptedTranscriber(["[18][S01]结尾[19]", "[18][S01]结尾[19]"])
        session = RealtimeSession(
            _config(),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1"),
            embedder=ThreadRecordingEmbedder(),
        )
        session.push_audio(_speech(20.0))
        self._run(session.run_pending())

        async def close_inside_the_loop():
            loop_ident = threading.get_ident()
            events = await session.close()
            return loop_ident, events

        loop_ident, events = self._run(close_inside_the_loop())

        self.assertIn("committed", [event["type"] for event in events])
        self.assertEqual(len(ThreadRecordingEmbedder.idents), 1)
        self.assertNotEqual(ThreadRecordingEmbedder.idents[0], loop_ident)

    def test_push_after_close_retains_no_audio(self):
        """关闭后再推流不得被会话保留。

        这个守卫没有公开可观测的症状：run_pending/close 都以 _closed 短路，committed
        也不变。它唯一的后果是会话不再持有音频（内存），所以只能断言内部缓冲。
        """
        transcriber = ScriptedTranscriber([])
        session = self._session(transcriber)
        self._run(session.close())

        session.push_audio(_speech(8.0))

        self.assertEqual(session.committed, [])
        self.assertEqual(session._buffer.total_seconds, 0.0)


class FailedCommitTest(unittest.TestCase):
    """定稿（声纹嵌入 + 落盘）抛异常时，水位线不得越过没有落盘的内容。

    ``Stitcher.ingest`` / ``flush`` 会**先**推进水位线、**后**由调用方定稿。定稿失败
    而水位线已经越过这些段落时，后续窗口会按"已定稿"或"跨水位线"把它们丢掉——那不是
    暂时性丢失，而是永久丢失，且没有任何重试能救回来。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def _session(self, transcriber, embedder) -> RealtimeSession:
        return RealtimeSession(
            _config(),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1"),
            embedder=embedder,
        )

    def test_a_failed_commit_does_not_let_the_watermark_pass_the_content(self):
        transcriber = ScriptedTranscriber(["[1][S01]你好[2]", "[1][S01]你好[2]"])
        session = self._session(transcriber, FlakyEmbedder())
        session.push_audio(_speech(8.0))

        first = asyncio.run(session.run_pending())
        self.assertEqual([event["type"] for event in first], ["error", "status"])
        self.assertEqual(session.committed, [])

        # 第二窗是 [0, 14]，覆盖同一段音频。第一次的 ingest 已经把水位线推到 2.0；若不
        # 回滚，这段会命中"已定稿"规则被丢弃，此后任何窗口都救不回来。
        session.push_audio(_speech(6.0))
        second = asyncio.run(session.run_pending())

        committed = [
            seg
            for event in second
            if event["type"] == "committed"
            for seg in event["segments"]
        ]
        self.assertEqual([seg["text"] for seg in committed], ["你好"])
        self.assertEqual(
            [row["text"] for row in SessionStore.load_committed(self.runs, "s1")], ["你好"]
        )

    def test_a_failed_teardown_commit_keeps_the_tail_recoverable(self):
        """close() 的收尾定稿失败后，重试必须还能把临时区定稿。

        三个回复都相同，收尾那一窗只产出临时段，于是嵌入器唯一的一次调用来自 teardown
        的 flush（逐窗路径的 committed 是空的，_commit 直接返回）。若不回滚，重试时
        ingest 会把这段按"已定稿"丢掉、flush 返回空表，close() 顺利返回、内容静默消失。
        """
        transcriber = ScriptedTranscriber(["[18][S01]结尾[19]"] * 3)
        session = self._session(transcriber, FlakyEmbedder())
        session.push_audio(_speech(20.0))
        asyncio.run(session.run_pending())

        with self.assertRaises(RuntimeError):
            asyncio.run(session.close())
        self.assertFalse(session.closed)

        events = asyncio.run(session.close())

        committed = [
            seg
            for event in events
            if event["type"] == "committed"
            for seg in event["segments"]
        ]
        self.assertEqual([seg["text"] for seg in committed], ["结尾"])
        self.assertEqual(
            [row["text"] for row in SessionStore.load_committed(self.runs, "s1")], ["结尾"]
        )


class CommittedSegmentContractTest(unittest.TestCase):
    """``transcript.jsonl`` 的字段形状是跨阶段契约：阶段二的 HTTP API、导出桥和前端
    都直接读它，而字段名本身并不说明 ``speaker`` 存的是 id、``speaker_name`` 才是显示名。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def test_to_dict_emits_the_seven_keys_and_speaker_holds_the_id(self):
        class Unembeddable:
            """声纹嵌入返回 None，于是段落落到 U00/未知 这条路径上——id 与显示名不同，
            互换两个键名或改掉任一个都会打挂下面的断言。"""

            embedding_dim = 2

            def embed(self, audio, sample_rate):
                return None

        transcriber = ScriptedTranscriber(["[1][S01]你好[2]"])
        session = RealtimeSession(
            _config(),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1"),
            embedder=Unembeddable(),
        )
        session.push_audio(_speech(8.0))

        events = asyncio.run(session.run_pending())

        committed = next(event for event in events if event["type"] == "committed")
        row = committed["segments"][0]
        self.assertEqual(
            set(row),
            {"id", "start", "end", "speaker", "speaker_name", "text", "speaker_confident"},
        )
        self.assertEqual(row["speaker"], "U00")
        self.assertEqual(row["speaker_name"], "未知")
        self.assertIs(row["speaker_confident"], False)
        self.assertEqual(row["id"], "seg-1")

        # 真正被写进 transcript.jsonl 的那一行与事件里的字典是同一份契约。
        self.assertEqual(SessionStore.load_committed(self.runs, "s1"), [row])


class SegmentAudioTest(unittest.TestCase):
    def test_segment_starting_before_the_window_returns_none(self):
        audio = np.zeros(16000, dtype=np.float32)
        seg = Segment(start=18.0, end=30.0, speaker="S01", text="跨窗", window_id=1)

        self.assertIsNone(segment_audio(audio, 20.0, seg, 16000))

    def test_segment_inside_the_window_is_sliced_by_offset(self):
        audio = np.arange(32000, dtype=np.float32)
        seg = Segment(start=1.0, end=2.0, speaker="S01", text="x", window_id=0)

        sliced = segment_audio(audio, 0.0, seg, 16000)

        self.assertEqual(sliced.size, 16000)
        self.assertEqual(sliced[0], 16000.0)


class SilenceGateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def _session(self, transcriber, **kwargs) -> RealtimeSession:
        # 注意不要把 silence_gate 同时写死再通过 **kwargs 传一次——那会是重复关键字。
        params = {"silence_gate": True}
        params.update(kwargs)
        return RealtimeSession(
            _config(**params),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1"),
        )

    def _run(self, coro):
        return asyncio.run(coro)

    def test_silent_window_skips_inference(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 3)
        session = self._session(transcriber)
        session.push_audio(_silence(10.0))

        events = self._run(session.run_pending())

        self.assertEqual(transcriber.window_seconds, [])
        status = next(event for event in events if event["type"] == "status")
        self.assertEqual(status["state"], "running")

    def test_silent_window_still_advances_the_schedule(self):
        """跳过后必须推进 last_run_sec，否则每次轮询都在重算同一个窗口。"""
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 3)
        session = self._session(transcriber)
        session.push_audio(_silence(10.0))
        self._run(session.run_pending())

        session.push_audio(_silence(2.0))
        events = self._run(session.run_pending())

        self.assertEqual(events, [])
        self.assertEqual(transcriber.window_seconds, [])

    def test_speech_after_silence_is_transcribed(self):
        transcriber = ScriptedTranscriber(["[1][S01]恢复[2]"] * 3)
        session = self._session(transcriber)
        session.push_audio(_silence(10.0))
        self._run(session.run_pending())

        session.push_audio(_speech(10.0))
        events = self._run(session.run_pending())

        self.assertEqual(len(transcriber.window_seconds), 1)
        self.assertIn("committed", [event["type"] for event in events])

    def test_close_runs_the_final_window_even_when_it_is_silent(self):
        # close() 刻意**不**经过静音门控：收尾那一窗是为了让临时区拿到最新结果。
        # 构造一个整段安静的收尾窗口，就能从转写器的调用次数上直接看出来——门控若
        # 在这里生效，最后这一窗根本不会被推理。
        #
        # 时序：前 8 秒有语音（首窗 [0,8] 出一段临时段），之后 22 秒全静音，于是
        # total = 30 时收尾窗口是 [10, 30]，整段落在这段静音里。
        transcriber = ScriptedTranscriber(["[6][S01]旧[7]", "[1][S01]新[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))
        self._run(session.run_pending())
        calls_after_first_run = len(transcriber.window_seconds)

        session.push_audio(_silence(22.0))
        self._run(session.close())

        self.assertEqual(len(transcriber.window_seconds), calls_after_first_run + 1)

    def test_gate_can_be_disabled(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 3)
        session = self._session(transcriber, silence_gate=False)
        session.push_audio(_silence(10.0))

        self._run(session.run_pending())

        self.assertEqual(len(transcriber.window_seconds), 1)

    def test_status_event_reports_gated_windows(self):
        """跳过必须可观测：状态事件在门控下始终报 running，与健康会话无从区分。

        没有这两个字段，"整场会议被误判成静音、一次推理都没跑"和正常会话发出的事件
        完全一样。
        """
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 3)
        session = self._session(transcriber)
        session.push_audio(_silence(10.0))

        events = self._run(session.run_pending())

        self.assertEqual(transcriber.window_seconds, [])
        status = next(event for event in events if event["type"] == "status")
        self.assertEqual(status["gated_windows"], 1)
        self.assertAlmostEqual(status["gated_sec"], 10.0, places=2)

    def _drive_gated(
        self, total_seconds: float, *, block: float | None = None, late_after: int | None = None, **kwargs
    ) -> tuple[list[tuple[float, float]], float]:
        """按 poll_interval 轮询、按 block 的粒度推流，返回已执行窗口区间与总时长。

        **两个节奏必须解耦**：决策点落在以执行结果为锚的网格上，而网格步长由
        ``total_seconds`` 每次跳多少决定，不是由 hop 决定。按 poll_interval 推流会让两者
        重合（步长恰好是 hop 的整数倍），把"网格被拉长"造成的空隙掩盖掉——那是上一版
        测试看不出来的原因。block 默认等于 poll_interval；late_after=k 表示第 k 次推流后
        不轮询（一次迟到的轮询），于是下一次轮询晚了一个间隔。

        末尾会调用 close()：收尾窗口的终点恰好是 total，于是"覆盖到会话末尾"变成一条可以
        直接断言的性质（见 _assert_covers_the_whole_session）。
        """
        config = _config(silence_gate=True, **kwargs)
        step = config.poll_interval
        block = step if block is None else block
        session = SpanRecordingSession(
            config,
            transcriber=ScriptedTranscriber([]),
            store=SessionStore(self.runs, "s1"),
        )
        pushed = 0.0
        pushes = 0
        while pushed < total_seconds - 1e-9:
            session.push_audio(_silence(block))
            pushed += block
            pushes += 1
            if late_after is not None and pushes == late_after:
                continue
            # 一个 block 的时长里该轮到几次轮询就轮询几次（没有新音频时的轮询是空操作，
            # 但保持与生产的 poll_interval 节奏一致）。
            for _ in range(max(1, int(round(block / step)))):
                self._run(session.run_pending())
        self._run(session.close())
        return session.spans, pushed

    def _assert_covers_the_whole_session(
        self, spans: list[tuple[float, float]], total: float
    ) -> None:
        """断言已执行窗口的并集**覆盖 [0, total] 且没有空隙**，一次报出全部空隙。

        空隙是不可恢复的：后续窗口的左边界只会更晚，永远够不到它。报告全部而不是只报
        第一个，是为了让"每次执行之后都留一条缝"的失败形态一次看清。

        前提是音频还在环形缓冲里。前端已经被淘汰时（驱动停摆期间音频仍在到达）任何窗口
        都读不到那一段——那不是这个断言能覆盖的情形，由
        ``test_a_front_that_fell_out_of_the_buffer_does_not_stall_the_session`` 单独钉住
        "会话还能继续跑"。
        """
        self.assertTrue(spans, "一次推理都没跑")
        holes: list[tuple[float, float]] = []
        reached = 0.0
        for start, end in spans:
            if start > reached + 1e-9:
                holes.append((round(reached, 3), round(start, 3)))
            reached = max(reached, end)
        if reached < total - 1e-9:
            holes.append((round(reached, 3), round(total, 3)))
        self.assertEqual(holes, [], f"这些区间没有任何窗口覆盖: {holes}")
        # 覆盖性是主断言，这条只防"门控退化成几乎不跑"。放在最后：否则它会先失败，把
        # 真正想报的空隙盖住。
        self.assertGreaterEqual(len(spans), 3, "门控把绝大部分推理都跳过了")

    def test_gate_preserves_reachability_at_the_poll_cadence(self):
        """按文档节奏（0.5 秒一块、0.5 秒一轮询）驱动时，覆盖是完整的。

        实测：已执行窗口 [0,18]、[18,38]、[38,58]，收尾窗口 [40,60]——并集覆盖 [0,60]。
        """
        spans, total = self._drive_gated(60.0)

        self._assert_covers_the_whole_session(spans, total)

    def test_gate_preserves_reachability_when_hop_does_not_divide_the_window(self):
        """window=20 / hop=7：window % hop != 0 时网格步长更难对齐。

        实测：已执行窗口 [0,15]、[9,29]、[23,43]、[37,57]，收尾窗口 [40,60]；相邻两两
        相接或重叠，并集覆盖 [0,60]。
        """
        spans, total = self._drive_gated(60.0, hop=7.0)

        self._assert_covers_the_whole_session(spans, total)

    def test_reachability_when_audio_arrives_in_larger_blocks(self):
        """音频按 1.5 秒一块到达（WebSocket 传输的常态）时，覆盖仍然完整。

        到达粒度会把决策网格的步长拉到 hop 以上（1.5 秒一块、hop=5 时步长是 6 秒），
        于是决策自己算出的左边界会落到上一次执行窗口的右边之后。**修复前实测空隙
        (0.0, 1.0)**：首个执行窗口是 [1.0, 21.0]，而后面每个窗口的左边界都更晚，
        开头那一秒永远没被读到。修复后首个窗口是 [0.0, 21.0]。
        """
        spans, total = self._drive_gated(60.0, block=1.5)

        self._assert_covers_the_whole_session(spans, total)

    def test_reachability_when_a_poll_arrives_late(self):
        """一次迟到的轮询（隔了两个间隔）不得留下永久空隙。

        **修复前实测空隙 (18.0, 18.5)**：首窗在 total=18.0 执行；第 76 次推流后那次轮询
        迟到，于是一直跳过的主循环把下一次执行推到 total=38.5，其左边界 18.5 落在上一次
        的右边界 18.0 之后。修复后该窗起点被钳回 18.0（实测 [18.0, 38.5]）。
        """
        spans, total = self._drive_gated(60.0, late_after=76)

        self._assert_covers_the_whole_session(spans, total)

    def test_a_front_that_fell_out_of_the_buffer_does_not_stall_the_session(self):
        """前端被淘汰时仍要能跑起来，不能永远卡在"请求已淘汰的音频"上。

        起点钳位让窗口往回收，收得太远（驱动停摆期间音频仍在到达，把保留头推到前端之后）
        时 slice 会返回 None；此时必须退回决策自己的窗口，否则每次轮询都会再问一次同一段
        已淘汰的音频，会话永远不再推理（实测退化形态：一次推理都不再发生）。

        容量 30 秒、40 秒一次性到达两次：第一次决策的 [0,40] 已经读不到（保留头在 10）。
        这里钉的是**会话还能继续**——total=80 时那次决策仍要真的执行，随后收尾窗口也要
        执行，两次都覆盖到末尾（实测 [(60.0, 80.0), (60.0, 80.0)]）。

        注意这条钉的是"还能跑"，不是"覆盖完整"：被淘汰的那段音频任何窗口都读不到，而
        回退到决策自己的窗口会让 (保留头, window 左边界) 这段**仍可读**的音频留在原地
        ——这是驱动停摆超过缓冲容量这种病态情形下的既有行为，不是这个修复引入的。
        """
        spans, total = self._drive_gated(80.0, block=40.0, buffer_capacity=30.0)

        self.assertEqual(spans, [(60.0, 80.0), (60.0, 80.0)])


class _StubEmbedder:
    """确定性向量，够让说话人表真的建起来。"""

    embedding_dim = 2

    def embed(self, audio, sample_rate):
        return np.array([1.0, 0.0], dtype=np.float32)


class PublicControlSurfaceTest(unittest.TestCase):
    """服务层不再伸进私有属性——这些就是它要用的那组公开入口。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def _session(self, **kwargs) -> RealtimeSession:
        return RealtimeSession(
            _config(),
            transcriber=ScriptedTranscriber(["[1][S01]你好[2]"] * 4),
            store=SessionStore(self.runs, "s1"),
            **kwargs,
        )

    def _run(self, coro):
        return asyncio.run(coro)

    def test_prompt_can_be_read_and_replaced(self):
        session = self._session()
        self.assertTrue(session.prompt)

        session.set_prompt("新 prompt")

        self.assertEqual(session.prompt, "新 prompt")

    def test_setting_an_empty_prompt_keeps_the_current_one(self):
        session = self._session()
        before = session.prompt

        session.set_prompt("")

        self.assertEqual(session.prompt, before)

    def test_speakers_is_empty_without_an_embedder(self):
        self.assertEqual(self._session().speakers(), [])

    def test_renaming_an_unknown_speaker_raises(self):
        with self.assertRaises(KeyError):
            self._session().rename_speaker("S99", "谁")

    def test_reassigning_an_unknown_segment_returns_none(self):
        self.assertIsNone(self._session().reassign_speaker("seg-999", "S01"))

    def test_reassigning_rewrites_the_stored_row(self):
        session = self._session()
        session.push_audio(_speech(8.0))
        self._run(session.run_pending())
        segment_id = session.committed[0].id

        updated = session.reassign_speaker(segment_id, "U99")

        self.assertEqual(updated.speaker_id, "U99")
        rows = SessionStore.load_committed(self.runs, "s1")
        self.assertEqual([row["speaker"] for row in rows], ["U99"])

    def test_reassigning_uses_the_renamed_display_name(self):
        session = self._session(embedder=_StubEmbedder())
        session.push_audio(_speech(8.0))
        self._run(session.run_pending())
        segment = session.committed[0]
        session.rename_speaker(segment.speaker_id, "张总")

        updated = session.reassign_speaker(segment.id, segment.speaker_id)

        self.assertEqual(updated.speaker_name, "张总")

class OverloadedStatusTest(unittest.TestCase):
    """spec §4.7：落后超过 `3 * W` 就要报告积压，前端据此提示"算力跟不上"。

    只报 `lag_sec` 是不够的——那个数在正常稳态下也有 `tail + hop` 那么大（默认约 11 秒），
    用户没法从它看出"这是正常的"还是"算力跟不上了"。所以服务端要把判断做完。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def _status(self, session, events):
        return [event for event in events if event["type"] == "status"][-1]

    def _run(self, session):
        return asyncio.run(session.run_pending())

    def _drive(self, session, *spans, chunk: float = 5.0):
        """按 `chunk` 秒一小块喂音频，每块之后轮询一次——真实 driver 就是这个节奏。

        不能一次性 `push_audio(70 秒)`：窗口策略取的是"当前能取的最新窗口"，一次灌完
        只会得到**一个**覆盖整个缓冲的大窗口，"距上次执行攒够 window - hop"这条门控
        条件永远不成立，测出来的东西和真机行为不是一回事。
        """
        events = []
        for kind, seconds in spans:
            remaining = seconds
            while remaining > 1e-9:
                step = min(chunk, remaining)
                session.push_audio(_silence(step) if kind == "silence" else _speech(step))
                events.extend(self._run(session))
                remaining -= step
        return events

    def test_a_healthy_session_is_not_reported_overloaded(self):
        # 8 秒音频、全定稿：落后 = 8 - 2 = 6 秒，远低于 3 * 20
        session = RealtimeSession(
            _config(),
            transcriber=ScriptedTranscriber(["[1][S01]a[2]"] * 4),
            store=SessionStore(self.runs, "s1"),
        )
        session.push_audio(_speech(8.0))

        status = self._status(session, self._run(session))

        self.assertFalse(status["overloaded"])
        self.assertLess(status["lag_sec"], 3 * 20.0)

    def test_a_session_far_behind_is_reported_overloaded(self):
        # 70 秒音频、一段都没定稿：落后 70 秒 > 3 * 20
        session = RealtimeSession(
            _config(max_consecutive_failures=9),
            transcriber=ScriptedTranscriber([""] * 20),
            store=SessionStore(self.runs, "s2"),
        )
        session.push_audio(_speech(70.0))

        status = self._status(session, self._run(session))

        self.assertTrue(status["overloaded"])
        self.assertGreater(status["lag_sec"], 3 * 20.0)

    def test_the_threshold_follows_the_configured_window(self):
        # 同一个 40 秒落后：window=20 时不算（3*20=60），window=10 时算（3*10=30）
        session = RealtimeSession(
            _config(window=10.0, hop=2.5, min_first_window=4.0),
            transcriber=ScriptedTranscriber([""] * 20),
            store=SessionStore(self.runs, "s3"),
        )
        session.push_audio(_speech(40.0))

        status = self._status(session, self._run(session))

        self.assertTrue(status["overloaded"])

    def test_a_long_silence_is_not_reported_overloaded(self):
        """静音不是积压：没人说话时没有内容可提交，定稿水位线本来就推不动。

        真机上量到过这个假阳性——一次 127 秒的真实录音里有一段被门控的静音，`lag` 一路
        涨到 46.2 秒，离 `3 * 20` 只差 14 秒。再长一点，一台**完全空闲**的 GPU 就会被报成
        "算力跟不上"。所以判断积压时要把这段扣掉；`lag_sec` 本身不动，前端显示的是用户
        真正感受到的延迟，那个数该涨就得涨。
        """
        session = RealtimeSession(
            _config(silence_gate=True),
            transcriber=ScriptedTranscriber([""] * 20),
            store=SessionStore(self.runs, "s4"),
        )
        status = self._status(session, self._drive(session, ("silence", 70.0)))

        self.assertGreater(status["gated_windows"], 0, "这段音频本来就该被门控跳过")
        self.assertGreater(status["lag_sec"], 3 * 20.0, "lag 确实越过了阈值——这正是要区分开的那件事")
        self.assertFalse(status["overloaded"], "没人说话不等于算力跟不上")

    def test_a_gated_silence_does_not_hide_a_real_backlog(self):
        """扣掉静音不能顺手把真正的积压也扣掉：静音之后还压着 80 秒语音，必须报。"""
        session = RealtimeSession(
            _config(silence_gate=True),
            transcriber=ScriptedTranscriber([""] * 40),
            store=SessionStore(self.runs, "s5"),
        )
        status = self._status(session, self._drive(session, ("silence", 40.0), ("speech", 80.0)))

        self.assertGreater(status["gated_windows"], 0)
        self.assertGreater(status["lag_sec"], 3 * 20.0)
        self.assertTrue(status["overloaded"], "静音之后还积压着大量语音，这才是算力跟不上")

    def test_a_window_of_digital_silence_commits_nothing(self):
        """覆盖性保证会逼出静音窗，而模型会在纯静音上幻觉出一整段话。

        真机上量到过：一段 `rms`/`peak` 全为 0 的窗口产出了 "I'm sorry, I can't assist
        with that request…（Qwen 的拒绝话术）" 并被当成 10 秒定稿写进会议纪要，还顺手造了
        一个说话人。没有语音就没有内容可提交——不管模型多自信地说了什么。
        """
        session = RealtimeSession(
            _config(silence_gate=True),
            transcriber=ScriptedTranscriber(
                ["[1][S01]I'm sorry, I can't assist with that request.[2]"] * 30
            ),
            store=SessionStore(self.runs, "s6"),
        )

        events = self._drive(session, ("silence", 40.0))

        statuses = [event for event in events if event["type"] == "status"]
        self.assertEqual(
            [event for event in events if event["type"] == "committed"], [], "静音窗不该有任何定稿"
        )
        # 确认这一轮真的动过（覆盖规则逼出的执行 + 门控跳过），否则这条测试可能只是
        # "什么都没跑"就通过了。
        self.assertGreater(
            statuses[-1]["gated_windows"] + statuses[-1]["last_window_ms"], 0
        )
        # 静音的落后不算积压：与门控窗走同一套扣除。
        self.assertFalse(statuses[-1]["overloaded"])



if __name__ == "__main__":
    unittest.main()

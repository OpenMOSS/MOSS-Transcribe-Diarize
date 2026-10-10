"""Orchestration: buffer audio, run windows, commit stable segments."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace

import numpy as np

from moss_transcribe_diarize.prompts import DEFAULT_PROMPT

from .buffer import AudioRingBuffer
from .config import RealtimeConfig
from .energy import is_silent
from .speaker import SpeakerEmbedder, SpeakerGallery
from .stitch import Segment, Stitcher
from .store import SessionStore
from .transcriber import WindowTranscriber
from .window import WindowPolicy


@dataclass(frozen=True, slots=True)
class CommittedSegment:
    id: str
    start: float
    end: float
    speaker_id: str
    speaker_name: str
    text: str
    confident: bool

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker_id,
            "speaker_name": self.speaker_name,
            "text": self.text,
            "speaker_confident": self.confident,
        }


def segment_audio(
    window_audio: np.ndarray,
    window_start: float,
    seg: Segment,
    sample_rate: int,
) -> np.ndarray | None:
    """按窗口内偏移切出段落音频。

    刻意不从环形缓冲重新取——缓冲可能已经回绕并淘汰了这段音频，而窗口副本
    一定是完整的。

    段落起点早于 ``window_start`` 时返回 ``None``：这段音频不在这份窗口副本里，按偏移
    切只能拿到一段被左边界截断的音频，用它算出的声纹会把说话人认错，而且此后一直认错。
    宁可交回 ``None``，让调用方走"未知说话人"路径。这条路径可达——``close()`` 的 flush
    可能拿到**前一个窗口**留下的临时段，而收尾窗口的起点比它晚。
    """
    if seg.start < window_start:
        return None
    total = int(window_audio.size)
    start = int(round((seg.start - window_start) * sample_rate))
    end = int(round((seg.end - window_start) * sample_rate))
    start = max(0, min(start, total))
    end = max(start, min(end, total))
    if end <= start:
        return None
    return window_audio[start:end]


class RealtimeSession:
    """一个会话的全部状态与行为。不负责网络，只吐事件。

    **调用契约（并发）**：本类不是线程安全的。调用方必须先 ``await`` ``run_pending``
    到完成，再调用 ``close()``——两者不得同时在飞。驱动的正常写法是每 ``poll_interval``
    秒 ``await session.run_pending()`` 一次；若写成
    ``asyncio.create_task(session.run_pending())`` 之后又 ``await session.close()``，
    两个窗口就会跑在两个 executor 工作线程里，同时改 ``Stitcher`` 的水位线与临时快照、
    ``SpeakerGallery`` 的条目、``_segment_counter`` 和 ``SessionStore``，后果是重复定稿
    或错乱的临时区——而且不会抛任何异常。
    """

    def __init__(
        self,
        config: RealtimeConfig,
        *,
        transcriber: WindowTranscriber,
        store: SessionStore,
        embedder: SpeakerEmbedder | None = None,
        prompt: str = DEFAULT_PROMPT,
    ):
        self.config = config
        self._transcriber = transcriber
        self._store = store
        self._prompt = prompt or DEFAULT_PROMPT
        self._buffer = AudioRingBuffer(config.buffer_capacity, config.sample_rate)
        self._policy = WindowPolicy(
            window=config.window,
            hop=config.hop,
            min_first_window=config.min_first_window,
        )
        self._stitcher = Stitcher(window=config.window, tail=config.tail)
        self._gallery = SpeakerGallery(
            embedder,
            threshold=config.speaker_threshold,
            min_segment_sec=config.min_segment_sec,
            sample_rate=config.sample_rate,
        )
        self._last_run_sec: float | None = None
        # 已覆盖前端：上一次**真正执行过**的窗口的末尾（含失败的那次）。与 _last_run_sec
        # 的区别是它只在窗口真的跑过时推进——被静音门控跳过的窗口不动它。
        # 两个用途：下一个执行窗口的起点被钳到它之前（保证覆盖连续——前端被淘汰时的例外
        # 见 run_pending 的回退分支），以及门控用它判断"距上次执行多久了"。
        self._last_executed_end = 0.0
        self._gated_windows = 0
        self._gated_sec = 0.0
        # 被判定为静音、**且不会再产出内容**的音频推进到哪：门控跳过的窗口，以及被覆盖
        # 规则逼着执行、但执行出来也全是幻觉的静音窗口（见 _process_window）。静音没有
        # 内容可提交，定稿水位线永远推不过它，所以它是"这段落后其实没人说话"的证据。
        # 判积压时用它把静音从 lag 里扣掉（见 _status_event）。
        self._silent_until = 0.0
        self._window_running = False
        self._failures = 0
        self._window_id = 0
        self._segment_counter = 0
        self._last_window_sec = 0.0
        self._closed = False
        self._committed: list[CommittedSegment] = []

    @property
    def committed(self) -> list[CommittedSegment]:
        return list(self._committed)

    @property
    def closed(self) -> bool:
        return self._closed

    def push_audio(self, pcm: np.ndarray) -> None:
        if self._closed:
            return
        arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        self._buffer.append(arr)
        self._store.append_audio(arr)

    async def run_pending(self) -> list[dict]:
        if self._closed or self._window_running:
            return []
        total = self._buffer.total_seconds
        decision = self._policy.decide(total_seconds=total, last_run_sec=self._last_run_sec)
        if not decision.should_run:
            return []
        # 起点钳到"已覆盖前端"：窗口调度落在以执行结果为锚的网格上，而**网格步长并不等于
        # hop**——决策要等 poll_interval，音频又按到达块推进 total_seconds，两者都会把步长
        # 拉长，且会话事先无从知道拉长多少。于是决策自己算出的左边界可能落在上一次执行窗口
        # 的右边界之后，那段夹缝永远没有任何窗口覆盖得到。取两者**更早**的一个即可：只要前端
        # 仍在缓冲里，每一次执行的窗口都与上一次相接或重叠，覆盖从第一窗（起点恒为 0）起
        # 连续——与网格步长、轮询节奏、音频到达粒度都无关。前端被淘汰时见下面的回退分支。
        # 代价：这样被迫放行的窗口可以长于 window（多出不到一个步长）。
        start = min(decision.start_sec, self._last_executed_end)
        audio = self._buffer.slice(start, decision.end_sec)
        if audio is None:
            # 前端已经滑出环形缓冲（例如驱动停摆期间音频仍在到达，把它挤掉了）。退回决策
            # 自己的窗口：那一段任何窗口都读不到（不是这个钳位能救回来的），但会话必须还能
            # 继续跑——否则每次轮询都会再问一次同一段已淘汰的音频，永远跑不起来。
            # 这不是覆盖的兜底：回退之后，(保留头, 决策左边界) 之间**仍可读**的那段音频不会
            # 被任何窗口读到。这一条只在"停摆时长超过缓冲容量"这种病态情形下出现，代价与
            # 修复前相同（修复前也从决策左边界开始），由测试单独钉住"会话还能继续跑"。
            start = decision.start_sec
            audio = self._buffer.slice(start, decision.end_sec)
        if audio is None or audio.size == 0:
            # 没有可推理的音频。推进 last_run_sec，避免每次轮询都重试。
            self._last_run_sec = decision.end_sec
            return []
        if self.config.silence_gate and is_silent(
            audio,
            self.config.sample_rate,
            self.config.silence_rms_db,
            self.config.silence_frame_ratio,
        ):
            # 门控只决定**多久放行一次**；覆盖由上面的起点钳位保证，不靠这个判据（先前
            # 版本试图用它保证覆盖，那是错的：网格步长不可预知）。
            # 判据是"距上次执行已攒够 window - hop 秒"，所以相邻两次执行的实际间隔
            # **不小于** window - hop——它是**下界**而不是近似值。实测：默认配置 20 秒、
            # hop=7 时 14 秒。
            # 上界不是 window：网格步长可以大于 hop（音频按块到达、轮询自身也有间隔，会话
            # 事先无从知道步长），所以间隔最多是 window - hop 再加一个网格步长。实测：12 秒
            # 一块、window=20 / hop=5 时间隔 24 秒（超过 window），放行的窗口因此被拉长。
            if (
                decision.end_sec - self._last_executed_end
                <= self.config.window - self.config.hop
            ):
                # 整窗静音：跳过推理，但必须推进 last_run_sec，否则下次轮询会重算
                # 同一个窗口，退化成忙等。
                self._last_run_sec = decision.end_sec
                self._gated_windows += 1
                self._gated_sec += decision.end_sec - decision.start_sec
                self._silent_until = max(self._silent_until, decision.end_sec)
                return [self._status_event()]
        return await self._run_window(start, decision.end_sec, audio)

    async def close(self) -> list[dict]:
        if self._closed:
            return []
        events: list[dict] = []
        total = self._buffer.total_seconds
        window_audio: np.ndarray | None = None
        window_start = 0.0
        if total > 0:
            # 与 run_pending 同样的起点钳位：收尾窗口也要勾回已覆盖前端，否则会话末尾
            # 那段（前端到 window 左边之间）的音频没有任何窗口读到过。
            window_start = min(
                max(0.0, total - self.config.window), self._last_executed_end
            )
            window_audio = self._buffer.slice(window_start, total)
            if window_audio is None:
                # 同 run_pending：前端已被淘汰时退回决策自己的窗口，别让收尾整窗落空。
                window_start = max(0.0, total - self.config.window)
                window_audio = self._buffer.slice(window_start, total)
            if window_audio is not None and window_audio.size:
                # 用正常的 tail 再跑一次，让临时区拿到最新推理结果，再整段定稿。
                events.extend(await self._run_window(window_start, total, window_audio))
        loop = asyncio.get_running_loop()
        # 收尾同样是阻塞工作：_commit 里的声纹嵌入要跑模型，finalize 要关 WAV 并重写元
        # 信息。整段放进工作线程，理由与逐窗路径相同——否则收尾期间事件循环被占住，在
        # WebSocket 传输下整个服务这段时间都不响应。
        had_remaining, newly = await loop.run_in_executor(
            None, self._teardown, window_audio, window_start
        )
        if had_remaining:
            if newly:
                events.append({"type": "committed", "segments": [seg.to_dict() for seg in newly]})
            events.append({"type": "provisional", "segments": []})
        events.append({"type": "speaker", "speakers": self._gallery.speakers()})
        self._closed = True
        return events

    # --- 公开的转写与说话人控制面 ---
    #
    # 服务层（WebSocket 控制指令）需要改 prompt、改说话人显示名、改某一段的说话人。
    # 没有这组方法时它只能伸进 ``_prompt`` / ``_gallery`` / ``_committed`` 这些私有属性
    # ——阶段一为 ``_gallery.rename`` 破过一次例，那是欠债。这组方法就是把那笔债还掉：
    # 会话自己拥有它的转写与说话人表。

    @property
    def prompt(self) -> str:
        return self._prompt

    def set_prompt(self, prompt: str) -> None:
        """换掉后续窗口用的 prompt。

        只影响**之后**的窗口；已经定稿的内容不会被改写——这与"定稿永不回改"一致。
        """
        self._prompt = str(prompt or self._prompt)

    def speakers(self) -> list[dict]:
        return self._gallery.speakers()

    def rename_speaker(self, speaker_id: str, name: str) -> None:
        """给全局说话人一个显示名。未知 id 抛 ``KeyError``。"""
        self._gallery.rename(speaker_id, name)

    def reassign_speaker(self, segment_id: str, speaker_id: str) -> CommittedSegment | None:
        """把一条已定稿段落改到另一个说话人，并**立刻落盘**。

        不落盘的话重开会话就丢了——而导出读的正是那份文件。找不到该 id 返回 ``None``。
        """
        for index, item in enumerate(self._committed):
            if item.id != segment_id:
                continue
            updated = replace(
                item,
                speaker_id=speaker_id,
                speaker_name=self._gallery.display_name(speaker_id),
            )
            self._committed[index] = updated
            self._store.rewrite_committed(self._committed)
            return updated
        return None

    # --- 内部 ---

    def _teardown(
        self,
        window_audio: np.ndarray | None,
        window_start: float,
    ) -> tuple[bool, list[CommittedSegment]]:
        """会话收尾：定稿临时区、落盘、关闭录音文件。

        与 ``_process_window`` 一样整个在工作线程里跑。返回的两个值正是事件构造需要
        的：本次 flush 是否有内容（决定要不要补一个"临时区已清空"事件），以及新定稿
        的段落。

        `stitcher` / `gallery` / `store` / 段落计数器都**不是**线程安全的；写它们之所以
        安全，靠的是 ``RealtimeSession`` 文档里的调用契约——调用方先 await ``run_pending``
        到完成再调用 ``close()``，两者不并发。这条契约由调用方保证，代码本身不强制它。
        """
        before_until = self._stitcher.committed_until
        before_provisional = self._stitcher.provisional
        remaining = self._stitcher.flush()
        try:
            newly = self._commit(remaining, window_audio, window_start)
        except Exception:
            # flush() 已经推过水位线、清空了临时快照，但内容还没落盘（_commit 里才有
            # 声纹嵌入与追加）。不回滚的话，调用方重试 close() 时 ingest 会按"已定稿"
            # 把同一段丢掉——静默、且不可恢复。
            self._stitcher.restore_after_failed_commit(before_until, before_provisional)
            raise
        # finalize 刻意不在回滚范围内：_commit 已经成功，内容确实落盘了；这时回滚只会
        # 让下一次 close() 把同一段再定稿一遍。
        self._store.finalize(self._gallery.speakers())
        return bool(remaining), newly

    async def _run_window(self, start: float, end: float, audio: np.ndarray) -> list[dict]:
        window_id = self._window_id
        self._window_id += 1
        self._window_running = True
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        try:
            committed, provisional = await loop.run_in_executor(
                None, self._process_window, start, end, window_id, audio
            )
        except Exception as exc:
            self._failures += 1
            return [
                {
                    "type": "error",
                    "code": "transcribe_failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "failures": self._failures,
                },
                self._status_event(),
            ]
        finally:
            self._window_running = False
            self._last_run_sec = end
            # 无论成败都推进：失败的那一窗同样需要窗口重新覆盖它，静音门控的下限对
            # 它同样适用（见 run_pending 里的说明）。
            self._last_executed_end = end
        self._failures = 0
        self._last_window_sec = time.perf_counter() - started

        events: list[dict] = []
        if committed:
            events.append({"type": "committed", "segments": [seg.to_dict() for seg in committed]})
        events.append(
            {
                "type": "provisional",
                "segments": [
                    {
                        "start": round(seg.start, 3),
                        "end": round(seg.end, 3),
                        "speaker": seg.speaker,
                        "text": seg.text,
                    }
                    for seg in provisional
                ],
            }
        )
        events.append({"type": "speaker", "speakers": self._gallery.speakers()})
        events.append(self._status_event())
        return events

    def _process_window(
        self,
        start: float,
        end: float,
        window_id: int,
        audio: np.ndarray,
    ) -> tuple[list[CommittedSegment], list[Segment]]:
        raw = self._transcriber.transcribe_window(audio, prompt=self._prompt)
        if self.config.silence_gate and is_silent(
            audio,
            self.config.sample_rate,
            self.config.silence_rms_db,
            self.config.silence_frame_ratio,
        ):
            # 这个窗口是**覆盖性保证**逼着执行的（真正的静音早被门控跳过了），里面没有
            # 语音可转写。可模型不会说"没有"，它会在纯静音上幻觉出一整段话——实机上量到
            # 过一段 Qwen 的拒绝话术（"I'm sorry, I can't assist with that request…"）
            # 被当成 10 秒定稿写进会议纪要，还顺手造了个说话人。既然没有语音，结果就丢掉；
            # 窗口的覆盖照样算数（`last_executed_end` 由调用方推进，见 run_pending），
            # 只是它也算"这段没人说话"，积压里要按静音扣掉。
            raw = ""
            self._silent_until = max(self._silent_until, end)
        # ingest 会推进水位线并替换临时快照，但内容此时还没落盘（嵌入与追加都在
        # _commit 里）。先记下这两块状态，定稿失败时回滚——否则水位线已经越过这些
        # 段落，下一个窗口会按"已定稿"或"跨水位线"把它们丢掉，内容永久消失，且没有
        # 任何重试能救回来。
        before_until = self._stitcher.committed_until
        before_provisional = self._stitcher.provisional
        result = self._stitcher.ingest(
            window_start=start,
            window_end=end,
            raw_text=raw,
            window_id=window_id,
        )
        try:
            committed = self._commit(result.committed, audio, start)
        except Exception:
            # write_provisional 不在回滚范围内：_commit 成功就说明内容已经落盘，回滚
            # 只会让下一个窗口重复定稿。
            self._stitcher.restore_after_failed_commit(before_until, before_provisional)
            raise
        self._store.write_provisional(result.provisional)
        return committed, result.provisional

    def _commit(
        self,
        segments: list[Segment],
        window_audio: np.ndarray | None,
        window_start: float,
    ) -> list[CommittedSegment]:
        if not segments:
            return []
        if window_audio is None:
            audio_of = lambda seg: None  # noqa: E731
        else:
            audio_of = lambda seg: segment_audio(  # noqa: E731
                window_audio, window_start, seg, self.config.sample_rate
            )
        assignments = self._gallery.assign(segments, audio_of)

        emitted: list[CommittedSegment] = []
        for item in assignments:
            self._segment_counter += 1
            seg = item.segment
            emitted.append(
                CommittedSegment(
                    id=f"seg-{self._segment_counter}",
                    start=seg.start,
                    end=seg.end,
                    speaker_id=item.speaker_id,
                    speaker_name=self._gallery.display_name(item.speaker_id),
                    text=seg.text,
                    confident=item.confident,
                )
            )
        self._store.append_committed(emitted)
        self._committed.extend(emitted)
        return emitted

    def _status_event(self) -> dict:
        hop = self.config.hop
        buffered = self._buffer.total_seconds
        # lag 用"字幕落后实时多少秒"来定义，而不是"缓冲里有多少秒音频没跑过推理"。
        # 后者恒等于零：_last_run_sec 每次都被设成窗口末尾，所以那个差值没有信息量。
        # 定稿水位线才是用户真正感受到的延迟，它天然包含 window + tail 的开销。
        lag = max(0.0, buffered - self._stitcher.committed_until)
        degraded = self._failures >= self.config.max_consecutive_failures
        # 积压要扣掉"被门控跳过的静音"：那段音频没有内容可提交，水位线本来就推不动，
        # 把它算成落后会把"没人说话"报成"算力跟不上"。真机上量到过——127 秒的真实录音
        # 里一段被门控的静音把 lag 顶到 46.2 秒，离 `3 * W` 只差 14 秒；再长一点，一台
        # 完全空闲的 GPU 就会被报成跟不上。`lag_sec` 本身不动：前端显示的是用户真正
        # 感受到的延迟，该涨就得涨。
        gated_pending = max(0.0, self._silent_until - self._stitcher.committed_until)
        # 稳态下 lag 天然有 tail + hop 那么大，用户没法从它区分"这是正常的"与"算力跟不上"，
        # 所以判断在服务端做完：落后超过 3 个 window 才算真的跟不上（spec §4.7）。
        # 用 W 而不是固定秒数，是因为 W 越大单次推理越久，可容忍的落后也越多。
        overloaded = lag - gated_pending > 3 * self.config.window
        return {
            "type": "status",
            "state": "degraded" if degraded else "running",
            "buffered_sec": round(buffered, 2),
            "rtf": round(self._last_window_sec / hop, 3) if hop > 0 else 0.0,
            "last_window_ms": int(round(self._last_window_sec * 1000)),
            "lag_sec": round(lag, 2),
            "degraded": degraded,
            "overloaded": overloaded,
            # 静音门控跳过过多少：这是"被跳过"唯一的对外可见之处（状态仍是 running），
            # 否则一个持续把语音误判成静音的会话与健康会话在事件上完全一样。
            "gated_windows": self._gated_windows,
            "gated_sec": round(self._gated_sec, 2),
        }

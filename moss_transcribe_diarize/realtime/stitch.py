"""Turn one window's raw model output into window-normalised segments."""

from __future__ import annotations

from dataclasses import dataclass, replace

from moss_transcribe_diarize.transcript_parser import TranscriptStreamParser

MIN_SEGMENT_DURATION = 0.05


@dataclass(frozen=True, slots=True)
class Segment:
    """一个转写段落。所有时间都是相对会话开始的绝对秒。"""

    start: float
    end: float
    speaker: str
    text: str
    window_id: int


def parse_window_segments(raw_text: str, window_start: float, window_id: int) -> list[Segment]:
    """解析模型输出，把窗口内的局部时间换算成绝对时间。

    解析器（``TranscriptStreamParser``）会丢弃它无法解析的部分，所以多数格式偏离
    只损失对应片段、不会让整次推理作废。

    两个必须知道的例外，都源于解析器的 ``_after_end``：``[end]`` 之后若跟着非空白的
    杂散文本，它会把 ``[end]`` 折回正文并退回"读取正文"状态。

    1. 流尾的杂散文本：该片段永不闭合，``close()`` 也不吐出它——窗口的最后一段丢失。
    2. 中间位置的杂散文本：紧随其后的 ``[时间戳]`` 会被当成**本片段**的结束时间，
       于是本片段文字被污染（正文里混入 ``[1.5]垃圾`` 之类）、结束时间被顶到下一段的
       起点，而那个 ``[Sxx]`` 在 ``_READ_START`` 状态下解析失败被丢弃，下一段随之
       静默消失。

    本模块**不**去修补这两种情况，因为修补不了：解析器把"正文里本来就有的方括号"
    和"被折回的时间戳"处理成同一种结果（都走 ``_after_end``，后面跟的都是任意文本），
    所以任何"正文含方括号就丢弃"之类的判据都会误伤合法内容——正文里的方括号是
    ``transcript_parser`` 有测试固定下来的刻意保留行为。误伤的代价是静默丢内容，比
    被污染的段落更隐蔽。

    两种例外都由**下一个窗口重新覆盖该段音频**来恢复，属于暂时性丢失而非永久丢失。

    另：时间戳顺序颠倒时解析器同样不闭合该片段（``_read_end`` 只接受
    ``end >= start``），且 ``_parse_timestamp`` 不产生负值，所以这里不需要交换分支
    ——那会是一段永远执行不到的死代码。
    """
    parser = TranscriptStreamParser()
    local = list(parser.feed(raw_text))
    local.extend(parser.close())
    out: list[Segment] = []
    for item in local:
        start = window_start + float(item.start)
        end = window_start + float(item.end)
        out.append(
            Segment(
                start=start,
                end=end,
                speaker=item.speaker or "",
                text=item.text,
                window_id=window_id,
            )
        )
    return out


def clamp_to_window(seg: Segment, window_start: float, window_end: float) -> Segment | None:
    """把段落裁剪进窗口，或判定它应当被丢弃。

    跨出**左**边界的段落裁剪起点——模型偶尔会给出早于窗口起点的时间戳，裁剪比
    丢弃少丢内容。跨出**右**边界的段落直接丢弃——右边界就是"现在"，下一轮推理
    会重新看到它，而裁剪会让文字长度和时间跨度对不上。
    """
    if seg.end <= window_start or seg.start >= window_end:
        return None
    if seg.end > window_end:
        return None
    start = max(seg.start, window_start)
    if seg.end - start < MIN_SEGMENT_DURATION:
        return None
    if start == seg.start:
        return seg
    return replace(seg, start=start)


@dataclass(slots=True)
class StitchResult:
    committed: list[Segment]
    provisional: list[Segment]


class Stitcher:
    """把逐次窗口推理的结果，切成定稿区与临时区。

    ``committed_until`` 是一条只进不退的水位线：只有严格位于它之后的段落才会
    被定稿。靠它天然去重——相邻窗口的定稿区在时间上重叠，但同一段内容只会被
    定稿一次。
    """

    def __init__(self, *, window: float, tail: float):
        if window <= 0:
            raise ValueError("window must be positive")
        if not 0.0 <= tail < window:
            raise ValueError("tail must be in [0, window)")
        self.window = float(window)
        self.tail = float(tail)
        self._committed_until = 0.0
        self._provisional: list[Segment] = []

    @property
    def committed_until(self) -> float:
        return self._committed_until

    @property
    def provisional(self) -> list[Segment]:
        return list(self._provisional)

    def ingest(
        self,
        *,
        window_start: float,
        window_end: float,
        raw_text: str,
        window_id: int,
    ) -> StitchResult:
        candidates: list[Segment] = []
        for seg in parse_window_segments(raw_text, window_start, window_id):
            clamped = clamp_to_window(seg, window_start, window_end)
            if clamped is None:
                continue
            if clamped.end <= self._committed_until:
                continue
            if clamped.start < self._committed_until:
                # 跨水位线，几乎总是上一次推理已定稿内容的重叠部分。重复比少量
                # 丢失更刺眼，所以整段丢弃，且不推进水位线。
                continue
            candidates.append(clamped)
        candidates.sort(key=lambda item: (item.start, item.end))

        cutoff = window_end - self.tail
        committed = [seg for seg in candidates if seg.end <= cutoff]
        self._provisional = [seg for seg in candidates if seg.end > cutoff]
        if committed:
            # 取 max 而不是 committed[-1].end：候选按 (start, end) 排序，重叠段的最后一个
            # 未必是 end 最大的那个。少推进会让下一个窗口重复定稿已定稿的内容（用户可见
            # 的重复字幕行）。
            self._committed_until = max(
                self._committed_until, max(seg.end for seg in committed)
            )
        return StitchResult(committed=committed, provisional=list(self._provisional))

    def flush(self) -> list[Segment]:
        """把临时区里剩下的段落全部定稿。会话结束时调用。"""
        remaining = self._provisional
        self._provisional = []
        if remaining:
            self._committed_until = max(self._committed_until, max(seg.end for seg in remaining))
        return remaining

    def restore_after_failed_commit(self, committed_until: float, provisional: list[Segment]) -> None:
        """把水位线与临时快照回滚到调用方事先捕获的值。

        存在的理由只有一个：``ingest`` / ``flush`` 会**先**推进水位线、**后**由调用方
        走定稿路径（声纹嵌入 + 落盘）。那一步可能失败，而水位线一旦越过这些段落，后续
        窗口就会按"已定稿"或"跨水位线"把它们丢掉——内容永久消失，且没有任何重试能救回
        来。所以定稿失败时必须把这两块状态放回去，让下一个窗口重新覆盖同一段音频。

        约定：这两个参数应当是**同一次**成功的 ingest/flush 之前捕获的
        ``committed_until`` 与 ``provisional``（后者用属性取到的副本即可）。
        """
        self._committed_until = float(committed_until)
        self._provisional = list(provisional)

from __future__ import annotations

import unittest

from moss_transcribe_diarize.realtime.stitch import (
    MIN_SEGMENT_DURATION,
    Segment,
    Stitcher,
    clamp_to_window,
    parse_window_segments,
)


def _rawl(*triples) -> str:
    """把 (start, end, speaker, text) 四元组拼成模型输出格式。"""
    return "".join(f"[{s}][{sp}]{t}[{e}]" for s, e, sp, t in triples)


class ParseWindowSegmentsTest(unittest.TestCase):
    def test_shifts_local_times_to_absolute(self):
        segments = parse_window_segments("[0.5][S01]你好[1.5]", window_start=40.0, window_id=3)

        self.assertEqual(
            segments,
            [Segment(start=40.5, end=41.5, speaker="S01", text="你好", window_id=3)],
        )

    def test_parses_multiple_segments(self):
        raw = "[0.48][S01]Welcome[1.66][12.26][S02]Ready[13.81]"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual([s.speaker for s in segments], ["S01", "S02"])
        self.assertEqual([s.text for s in segments], ["Welcome", "Ready"])

    def test_whitespace_separated_noise_does_not_block_later_segments(self):
        # 杂散文本由空白分隔时走 _after_end 的 _pending_after_end 分支：下一个 [ 到来
        # 即干净地 emit 本段，后续片段照常产出。这是"偏离格式不作废整次推理"的良性
        # 一侧。
        raw = "垃圾开头 [0.5][S01]有效[1.5] [2.0][S02]也好[3.0]"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual([s.text for s in segments], ["有效", "也好"])

    def test_unseparated_noise_pollutes_a_segment_and_we_pass_it_through(self):
        # 没有空白分隔时，解析器把 [1.5] 折回正文，再把 [2.0] 当成该段的结束时间
        # （2.0 >= 0.5 故 _read_end 接受），于是本段变成
        # (0.5, 2.0, "S01", "有效[1.5]垃圾")——文字与结束时间双双被污染；而 [S02] 在
        # _READ_START 状态解析失败被 reset，"也好" 随之静默消失。
        #
        # 本模块原样透传，不做修补：解析器无法区分这种污染与它刻意保留的正文方括号
        # （见下面那条用例），任何"含括号就丢弃"的判据都会误伤合法内容。这段音频会由
        # 下一个窗口重新覆盖，所以是暂时性丢失。
        raw = "[0.5][S01]有效[1.5]垃圾[2.0][S02]也好[3.0]"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual([s.text for s in segments], ["有效[1.5]垃圾"])
        self.assertEqual(segments[0].end, 2.0)

    def test_brackets_inside_speech_are_preserved(self):
        # transcript_parser 有明确的契约（见 tests/test_transcript_parser.py 的
        # test_numeric_brackets_inside_text_are_preserved）：正文里的方括号是被刻意
        # 保留的，不是污染。任何"正文含方括号就丢弃"的判据都会误伤这条，所以这里钉住它。
        raw = "[0][S01]第[2024]年，编号[001]继续[4]"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual([s.text for s in segments], ["第[2024]年，编号[001]继续"])

    def test_trailing_prose_after_the_last_segment_drops_that_segment(self):
        # 流尾的杂散文本（_after_end 的第 1 种例外）：[end] 之后遇到非空白字符会把
        # [end] 折回正文并退回"读取正文"状态，于是该片段永不闭合、close() 也不吐出
        # 它。窗口的最后一段因此会丢；下一个窗口会重新覆盖这段音频，所以是暂时性丢失。
        raw = "这是模型的解释性文字 [0.5][S01]有效内容[1.5] 更多废话"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual(segments, [])

    def test_trailing_segment_without_end_is_dropped(self):
        raw = "[0.5][S01]完整[1.5][2.0][S02]缺尾巴"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual([s.text for s in segments], ["完整"])

    def test_inverted_timestamps_drop_the_segment(self):
        # 解析器只接受 end >= start；顺序颠倒时它把 [end] 折回正文、该片段不闭合，
        # 于是整段被丢弃。所以 parse_window_segments 里不需要交换分支——那会是一段
        # 永远执行不到的死代码。
        raw = "[3.0][S01]顺序反了[1.0]"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual(segments, [])

    def test_empty_text_yields_nothing(self):
        self.assertEqual(parse_window_segments("", window_start=0.0, window_id=0), [])


class ClampToWindowTest(unittest.TestCase):
    def test_segment_inside_window_is_unchanged(self):
        seg = Segment(start=5.0, end=7.0, speaker="S01", text="a", window_id=0)

        self.assertEqual(clamp_to_window(seg, 0.0, 20.0), seg)

    def test_segment_crossing_left_edge_is_truncated(self):
        seg = Segment(start=-1.0, end=3.0, speaker="S01", text="a", window_id=0)

        clamped = clamp_to_window(seg, 0.0, 20.0)

        self.assertEqual(clamped.start, 0.0)
        self.assertEqual(clamped.end, 3.0)
        self.assertEqual(clamped.text, "a")

    def test_segment_crossing_right_edge_is_dropped(self):
        seg = Segment(start=19.0, end=21.0, speaker="S01", text="a", window_id=0)

        self.assertIsNone(clamp_to_window(seg, 0.0, 20.0))

    def test_segment_entirely_before_window_is_dropped(self):
        seg = Segment(start=-5.0, end=-1.0, speaker="S01", text="a", window_id=0)

        self.assertIsNone(clamp_to_window(seg, 0.0, 20.0))

    def test_segment_entirely_after_window_is_dropped(self):
        seg = Segment(start=25.0, end=27.0, speaker="S01", text="a", window_id=0)

        self.assertIsNone(clamp_to_window(seg, 0.0, 20.0))

    def test_segment_shorter_than_minimum_after_clamping_is_dropped(self):
        seg = Segment(
            start=-1.0,
            end=MIN_SEGMENT_DURATION / 2.0,
            speaker="S01",
            text="a",
            window_id=0,
        )

        self.assertIsNone(clamp_to_window(seg, 0.0, 20.0))


class StitcherTest(unittest.TestCase):
    def test_commits_segments_older_than_tail(self):
        stitcher = Stitcher(window=20.0, tail=6.0)

        result = stitcher.ingest(
            window_start=0.0,
            window_end=20.0,
            raw_text=_rawl((1.0, 3.0, "S01", "早")),
            window_id=0,
        )

        self.assertEqual([s.text for s in result.committed], ["早"])
        self.assertEqual(result.provisional, [])
        self.assertEqual(stitcher.committed_until, 3.0)

    def test_holds_back_segments_inside_tail(self):
        stitcher = Stitcher(window=20.0, tail=6.0)

        result = stitcher.ingest(
            window_start=0.0,
            window_end=20.0,
            raw_text=_rawl((16.0, 19.0, "S01", "新")),
            window_id=0,
        )

        self.assertEqual(result.committed, [])
        self.assertEqual([s.text for s in result.provisional], ["新"])
        self.assertEqual(stitcher.committed_until, 0.0)

    def test_provisional_is_replaced_not_appended(self):
        stitcher = Stitcher(window=20.0, tail=6.0)
        stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((16.0, 19.0, "S01", "第一版")), window_id=0,
        )

        # 第二个窗口是 [5, 25]，raw 里的时间戳是**窗口局部**的，会被加上 window_start。
        # 局部 15.0-18.5 换算成绝对 20.0-23.5，落在 tail 区（> 25-6=19）内，所以是临时段。
        # 用它替换第一次的临时快照，而不是追加——这是 provisional 的语义。
        result = stitcher.ingest(
            window_start=5.0, window_end=25.0,
            raw_text=_rawl((15.0, 18.5, "S01", "第二版")), window_id=1,
        )

        self.assertEqual([s.text for s in result.provisional], ["第二版"])

    def test_already_committed_segment_is_not_recommitted(self):
        stitcher = Stitcher(window=20.0, tail=6.0)
        first = stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((1.0, 3.0, "S01", "早")), window_id=0,
        )
        self.assertEqual([s.text for s in first.committed], ["早"])
        self.assertEqual(stitcher.committed_until, 3.0)

        # 同一个窗口再跑一次，模型对同一段音频给出同样的结果。去重靠时间水位线，不靠
        # 文字比对：该段的 end 已等于 committed_until，整段落在水位线之前，丢弃。
        result = stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((1.0, 3.0, "S01", "早")), window_id=1,
        )

        self.assertEqual(result.committed, [])
        self.assertEqual(stitcher.committed_until, 3.0)

    def test_repeated_text_at_a_different_time_is_committed(self):
        # 去重靠时间水位线，**绝不比对文字**。同一句话在不同时刻说出来是正常的——
        # "嗯"、"对"、"是的" 这类会反复出现——按文字去重会把这些静默丢掉。
        stitcher = Stitcher(window=20.0, tail=6.0)
        first = stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((1.0, 3.0, "S01", "嗯")), window_id=0,
        )
        self.assertEqual([s.text for s in first.committed], ["嗯"])
        self.assertEqual(stitcher.committed_until, 3.0)

        # 第二段在窗口 [5, 25] 内的 6.0-8.0 秒处，换算成绝对时间 11.0-13.0，
        # 严格位于水位线 3.0 之后，所以必须定稿——尽管文字与上一段完全相同。
        second = stitcher.ingest(
            window_start=5.0, window_end=25.0,
            raw_text=_rawl((6.0, 8.0, "S01", "嗯")), window_id=1,
        )

        self.assertEqual([s.text for s in second.committed], ["嗯"])
        self.assertEqual(stitcher.committed_until, 13.0)

    def test_watermark_advances_past_overlapping_segments(self):
        # 候选按 (start, end) 排序，所以最后一个不一定是 end 最大的那个：这里
        # (1, 10) 排在 (2, 3) 前面。若水位线取 committed[-1].end，它只推到 3.0，
        # 3-10 这段已定稿内容会被下一个窗口再定稿一次——用户可见的重复字幕行。
        stitcher = Stitcher(window=20.0, tail=0.0)
        first = stitcher.ingest(
            window_start=0.0,
            window_end=20.0,
            raw_text=_rawl((1.0, 10.0, "S01", "长"), (2.0, 3.0, "S02", "短")),
            window_id=0,
        )
        self.assertEqual([s.text for s in first.committed], ["长", "短"])
        self.assertEqual(stitcher.committed_until, 10.0)

        second = stitcher.ingest(
            window_start=0.0,
            window_end=20.0,
            raw_text=_rawl((5.0, 7.0, "S01", "重叠区")),
            window_id=1,
        )

        self.assertEqual(second.committed, [])
        self.assertEqual(stitcher.committed_until, 10.0)

    def test_segment_straddling_the_watermark_is_dropped(self):
        stitcher = Stitcher(window=20.0, tail=6.0)
        stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((5.0, 10.0, "S01", "前半")), window_id=0,
        )

        result = stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((8.0, 12.0, "S01", "重叠")), window_id=1,
        )

        self.assertEqual(result.committed, [])
        self.assertEqual(stitcher.committed_until, 10.0)

    def test_out_of_range_timestamp_never_advances_the_watermark(self):
        stitcher = Stitcher(window=20.0, tail=6.0)

        result = stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((1.0, 3.0, "S01", "正常"), (5000.0, 9000.0, "S02", "未来")),
            window_id=0,
        )

        self.assertEqual([s.text for s in result.committed], ["正常"])
        self.assertEqual(stitcher.committed_until, 3.0)

    def test_later_window_can_still_commit_after_an_out_of_range_spike(self):
        stitcher = Stitcher(window=20.0, tail=6.0)
        stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((5000.0, 9000.0, "S02", "未来")), window_id=0,
        )

        result = stitcher.ingest(
            window_start=5.0, window_end=25.0,
            raw_text=_rawl((6.0, 9.0, "S01", "后续正常")), window_id=1,
        )

        self.assertEqual([s.text for s in result.committed], ["后续正常"])

    def test_committed_segments_are_returned_in_time_order(self):
        stitcher = Stitcher(window=20.0, tail=0.0)

        result = stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((8.0, 9.0, "S02", "后"), (1.0, 2.0, "S01", "前")),
            window_id=0,
        )

        self.assertEqual([s.text for s in result.committed], ["前", "后"])

    def test_flush_promotes_remaining_provisional(self):
        stitcher = Stitcher(window=20.0, tail=6.0)
        stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((16.0, 19.0, "S01", "末尾")), window_id=0,
        )

        flushed = stitcher.flush()

        self.assertEqual([s.text for s in flushed], ["末尾"])
        self.assertEqual(stitcher.flush(), [])
        self.assertEqual(stitcher.committed_until, 19.0)

    def test_rejects_tail_at_or_past_window(self):
        with self.assertRaises(ValueError):
            Stitcher(window=10.0, tail=10.0)


class StitcherUnparseableOutputTest(unittest.TestCase):
    def test_garbage_text_does_not_block_valid_segments(self):
        stitcher = Stitcher(window=20.0, tail=6.0)

        result = stitcher.ingest(
            window_start=0.0,
            window_end=20.0,
            raw_text="抱歉，我无法处理这段音频。" + _rawl((2.0, 4.0, "S01", "有效")),
            window_id=0,
        )

        self.assertEqual([s.text for s in result.committed], ["有效"])

    def test_fully_unparseable_output_yields_nothing(self):
        stitcher = Stitcher(window=20.0, tail=6.0)

        result = stitcher.ingest(
            window_start=0.0, window_end=20.0, raw_text="没有任何时间戳", window_id=0,
        )

        self.assertEqual(result.committed, [])
        self.assertEqual(result.provisional, [])
        self.assertEqual(stitcher.committed_until, 0.0)


if __name__ == "__main__":
    unittest.main()

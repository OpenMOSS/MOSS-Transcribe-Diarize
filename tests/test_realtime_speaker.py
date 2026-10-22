from __future__ import annotations

import unittest

import numpy as np

from moss_transcribe_diarize.realtime.speaker import (
    UNKNOWN_SPEAKER_ID,
    SpeakerGallery,
)
from moss_transcribe_diarize.realtime.stitch import Segment


class FakeEmbedder:
    """按预设的向量表返回嵌入，未登记的输入返回 None。"""

    embedding_dim = 3

    def __init__(self, mapping: dict[bytes, list[float]], *, default=None):
        self.mapping = {key: np.array(value, dtype=np.float32) for key, value in mapping.items()}
        self.default = None if default is None else np.array(default, dtype=np.float32)

    def embed(self, audio, sample_rate):
        key = np.asarray(audio, dtype=np.float32).tobytes()
        return self.mapping.get(key, self.default)


def _voice(value: float, *, samples: int = 16000) -> np.ndarray:
    """一段**可信赖长度**的假音频，用填充值区分身份。

    长度必须同时超过两个门槛：``min_segment_sec``（默认 0.4 秒 = 6400 样本，低于它连
    嵌入都不算）与 ``min_reliable_sec``（默认 1.0 秒 = 16000 样本，低于它算出的向量是
    噪声，不许开新身份）。默认给 1.0 秒，让"匹配/开新身份"这些测试跑在可信区间里。
    想测"太短"的两种分支，就显式传小 `samples` 或给 gallery 传小 `min_reliable_sec`。
    """
    return np.full(samples, value, dtype=np.float32)


def _seg(start: float, end: float, speaker: str, window_id: int, text: str = "x") -> Segment:
    return Segment(start=start, end=end, speaker=speaker, text=text, window_id=window_id)


class SpeakerGalleryTest(unittest.TestCase):
    def test_disabled_gallery_falls_back_to_local_labels(self):
        gallery = SpeakerGallery(None)

        self.assertFalse(gallery.enabled)
        assignments = gallery.assign([_seg(0.0, 1.0, "S02", 0)], lambda seg: None)

        self.assertEqual(assignments[0].speaker_id, "S02")
        self.assertFalse(assignments[0].confident)

    def test_first_speaker_gets_first_global_id(self):
        voice = _voice(1.0)
        gallery = SpeakerGallery(FakeEmbedder({voice.tobytes(): [1.0, 0.0, 0.0]}))

        assignments = gallery.assign([_seg(0.0, 1.0, "S02", 0)], lambda seg: voice)

        self.assertEqual(assignments[0].speaker_id, "S01")
        self.assertTrue(assignments[0].confident)

    def test_same_voice_across_windows_keeps_one_global_id(self):
        voice = _voice(1.0)
        gallery = SpeakerGallery(FakeEmbedder({voice.tobytes(): [1.0, 0.0, 0.0]}))
        audio_of = lambda seg: voice  # noqa: E731

        gallery.assign([_seg(0.0, 1.0, "S01", 0)], audio_of)
        assignments = gallery.assign([_seg(20.0, 21.0, "S01", 1)], audio_of)

        self.assertEqual(assignments[0].speaker_id, "S01")
        self.assertEqual(len(gallery.speakers()), 1)

    def test_different_voice_gets_a_new_global_id(self):
        first = _voice(1.0)
        second = _voice(2.0)
        embedder = FakeEmbedder(
            {first.tobytes(): [1.0, 0.0, 0.0], second.tobytes(): [0.0, 1.0, 0.0]}
        )
        gallery = SpeakerGallery(embedder)

        gallery.assign([_seg(0.0, 1.0, "S01", 0)], lambda seg: first)
        assignments = gallery.assign([_seg(20.0, 21.0, "S01", 1)], lambda seg: second)

        self.assertEqual(assignments[0].speaker_id, "S02")
        self.assertEqual(len(gallery.speakers()), 2)

    def test_local_labels_are_only_a_within_window_prior(self):
        """同一窗口里两个局部标签若声纹相同，应归到同一个全局说话人。"""
        voice = _voice(1.0)
        gallery = SpeakerGallery(FakeEmbedder({voice.tobytes(): [1.0, 0.0, 0.0]}))
        audio_of = lambda seg: voice  # noqa: E731

        assignments = gallery.assign(
            [_seg(0.0, 1.0, "S01", 0), _seg(2.0, 3.0, "S02", 0)], audio_of
        )

        self.assertEqual([a.speaker_id for a in assignments], ["S01", "S01"])

    def test_group_members_inherit_the_group_decision(self):
        """同窗口同局部标签的段落共享一次判定，包括太短而拿不到声纹的那段。"""
        # 两段都超过 min_segment_sec=0.1 的门槛（1600 样本），但 long_voice 更长，
        # 所以组代表是它；short_voice 的字节不在向量表里，一旦被选为组代表就会
        # 拿到 None，测试随即暴露选择逻辑写反了。
        long_voice = _voice(1.0, samples=2000)
        short_voice = _voice(2.0, samples=1700)
        gallery = SpeakerGallery(
            FakeEmbedder({long_voice.tobytes(): [1.0, 0.0, 0.0]}, default=None),
            min_segment_sec=0.1,
            # 这条测试钉的是"组代表怎么选"，不是"多长才算可信"，所以把可靠门槛一起压低，
            # 让 2000/1700 样本这种长度仍然走"开新身份"那条路。
            min_reliable_sec=0.01,
        )

        def audio_of(seg):
            return long_voice if seg.text == "long" else short_voice

        assignments = gallery.assign(
            [
                _seg(0.0, 5.0, "S03", 0, text="long"),
                _seg(6.0, 6.02, "S03", 0, text="short"),
            ],
            audio_of,
        )

        self.assertEqual([a.speaker_id for a in assignments], ["S01", "S01"])

    def test_same_local_label_in_different_windows_is_not_merged(self):
        # 分组的键是 (window_id, 局部标签)。局部标签只是**窗口内**的先验：窗口 A 的
        # S01 与窗口 B 的 S01 完全可以是两个人，按 seg.speaker 单独分组会把不同的人
        # 合并成一个全局说话人。
        #
        # 注意：当前唯一的调用方每个窗口调用 assign 一次，所以本次调用里的段共享同一个
        # window_id，这个键分量今天是**防御性**的。这条测试钉的是意图——将来若有人把键
        # 简化成 seg.speaker，应当是一次可见的决定，而不是一个静默的改动。
        first = _voice(1.0)
        second = _voice(2.0)
        gallery = SpeakerGallery(
            FakeEmbedder({first.tobytes(): [1.0, 0.0, 0.0], second.tobytes(): [0.0, 1.0, 0.0]})
        )

        assignments = gallery.assign(
            [
                _seg(0.0, 1.0, "S01", window_id=0),
                _seg(20.0, 21.0, "S01", window_id=1),
            ],
            lambda seg: first if seg.window_id == 0 else second,
        )

        self.assertEqual([a.speaker_id for a in assignments], ["S01", "S02"])
        self.assertEqual(len(gallery.speakers()), 2)

    def test_unembeddable_group_is_marked_unconfident(self):
        gallery = SpeakerGallery(FakeEmbedder({}), min_segment_sec=0.1)

        assignments = gallery.assign([_seg(0.0, 5.0, "S01", 0)], lambda seg: _voice(9.0))

        self.assertEqual(assignments[0].speaker_id, UNKNOWN_SPEAKER_ID)
        self.assertFalse(assignments[0].confident)

    def test_too_short_segment_is_not_embedded(self):
        # 100 样本远低于 min_segment_sec=1.0 对应的 16000 样本门槛
        voice = _voice(1.0, samples=100)
        gallery = SpeakerGallery(FakeEmbedder({voice.tobytes(): [1.0, 0.0, 0.0]}), min_segment_sec=1.0)

        assignments = gallery.assign([_seg(0.0, 0.2, "S01", 0)], lambda seg: voice)

        self.assertFalse(assignments[0].confident)
        # 只断言 confident 不够：过短分支若错成"返回局部标签"，confident 仍是 False。
        self.assertEqual(assignments[0].speaker_id, UNKNOWN_SPEAKER_ID)

    def test_unknown_speaker_appears_in_the_roster(self):
        gallery = SpeakerGallery(FakeEmbedder({}), min_segment_sec=0.1)

        gallery.assign([_seg(0.0, 5.0, "S01", 0)], lambda seg: _voice(9.0))

        roster = {item["id"]: item for item in gallery.speakers()}
        self.assertIn(UNKNOWN_SPEAKER_ID, roster)

    def test_rename_and_display_name(self):
        voice = _voice(1.0)
        gallery = SpeakerGallery(FakeEmbedder({voice.tobytes(): [1.0, 0.0, 0.0]}))
        gallery.assign([_seg(0.0, 1.0, "S01", 0)], lambda seg: voice)

        self.assertEqual(gallery.display_name("S01"), "S01")
        gallery.rename("S01", "张总")
        self.assertEqual(gallery.display_name("S01"), "张总")

    def test_rename_unknown_speaker_raises(self):
        gallery = SpeakerGallery(None)

        with self.assertRaises(KeyError):
            gallery.rename("S99", "谁")

    def test_centroid_update_keeps_a_consistent_voice_matched(self):
        """质心按样本数滑动平均更新，同一嗓音重复出现不应分裂出新的说话人。"""
        voice = _voice(1.0)
        gallery = SpeakerGallery(FakeEmbedder({voice.tobytes(): [1.0, 0.0, 0.0]}))
        audio_of = lambda seg: voice  # noqa: E731

        for index in range(5):
            assignments = gallery.assign([_seg(float(index) * 10, float(index) * 10 + 1, "S01", index)], audio_of)
            self.assertEqual(assignments[0].speaker_id, "S01")

        roster = gallery.speakers()
        self.assertEqual(len(roster), 1)
        self.assertEqual(roster[0]["samples"], 5)

    def test_a_short_unmatched_clip_does_not_mint_a_speaker(self):
        """半秒的短句匹配不上时，不许凭空造一个新说话人。

        真机上量到过：同一个真人在 127 秒里被拆成 S01–S04——"哈喽。"（0.50 秒）成了 S02、
        "Hello?"（0.72 秒）成了 S03、"呃。"（0.46 秒）成了 S04，全标着 confident。那几段
        与主说话人质心的相似度只有 0.18–0.48，彼此之间也只有 0.17–0.50：向量是噪声。
        """
        known = _voice(1.0)
        short_unknown = _voice(2.0, samples=8000)  # 0.5 秒，低于可靠门槛
        gallery = SpeakerGallery(
            FakeEmbedder(
                {known.tobytes(): [1.0, 0.0, 0.0], short_unknown.tobytes(): [0.0, 1.0, 0.0]}
            )
        )
        gallery.assign([_seg(0.0, 1.0, "S01", 0)], lambda seg: known)

        assignments = gallery.assign([_seg(20.0, 20.5, "S01", 1)], lambda seg: short_unknown)

        self.assertEqual(assignments[0].speaker_id, "S01", "归到最近的那个已有说话人")
        self.assertFalse(assignments[0].confident, "短句的判定不可信，必须如实标注")
        self.assertEqual([item["id"] for item in gallery.speakers()], ["S01"])
        # 关键：噪声没有进质心。进了的话它会把后面真正属于 S01 的语音全推开，而那一步
        # 是这套系统里最难恢复的。
        self.assertEqual(gallery.speakers()[0]["samples"], 1)

    def test_a_short_first_clip_does_not_establish_a_speaker(self):
        """空库里的第一句如果只有半秒，就报"未知"，而不是拿噪声立一个 S01。

        立了 S01 就等于把那条噪声向量设成全库参照：后面所有真正的语音都会因为与它不像
        而不断开新身份——这正是真机上"一个真人四个身份"的起点。
        """
        short = _voice(1.0, samples=8000)
        gallery = SpeakerGallery(FakeEmbedder({short.tobytes(): [1.0, 0.0, 0.0]}))

        assignments = gallery.assign([_seg(0.0, 0.5, "S01", 0)], lambda seg: short)

        self.assertEqual(assignments[0].speaker_id, UNKNOWN_SPEAKER_ID)
        self.assertFalse(assignments[0].confident)
        self.assertEqual([item["id"] for item in gallery.speakers()], [UNKNOWN_SPEAKER_ID])

    def test_a_long_clip_still_mints_a_speaker(self):
        """门槛只管短句：够长的音频照样能开新身份，别把功能一起关掉。"""
        first = _voice(1.0)
        second = _voice(2.0, samples=24000)  # 1.5 秒
        gallery = SpeakerGallery(
            FakeEmbedder({first.tobytes(): [1.0, 0.0, 0.0], second.tobytes(): [0.0, 1.0, 0.0]})
        )
        gallery.assign([_seg(0.0, 1.0, "S01", 0)], lambda seg: first)

        assignments = gallery.assign([_seg(20.0, 21.5, "S01", 1)], lambda seg: second)

        self.assertEqual(assignments[0].speaker_id, "S02")
        self.assertTrue(assignments[0].confident)

    def test_empty_input_returns_empty(self):
        gallery = SpeakerGallery(None)

        self.assertEqual(gallery.assign([], lambda seg: None), [])

if __name__ == "__main__":
    unittest.main()

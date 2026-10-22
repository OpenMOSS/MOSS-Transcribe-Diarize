"""Cross-window speaker identity via speaker embeddings."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .stitch import Segment

UNKNOWN_SPEAKER_ID = "U00"
UNKNOWN_SPEAKER_NAME = "未知"


@runtime_checkable
class SpeakerEmbedder(Protocol):
    """声纹嵌入。实现方负责把任意长度音频压成一个定长向量。"""

    embedding_dim: int

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
        """返回嵌入，或在这段音频无法可靠嵌入时返回 ``None``。"""


@dataclass(frozen=True, slots=True)
class Assignment:
    segment: Segment
    speaker_id: str
    confident: bool


@dataclass(slots=True)
class _GalleryEntry:
    id: str
    centroid: np.ndarray
    samples: int = 1
    name: str = ""


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0 or not np.isfinite(norm):
        return arr
    return arr / norm


SPEAKER_MODEL_CACHE = Path.home() / ".cache" / "mtd-speaker"
SPEAKER_MODEL_URL = "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/{}"

# 文件名 -> sha256。**锁定到具体产物并校验哈希**：声纹模型被换掉会静默破坏跨窗口
# 说话人一致性，而那种错误在转写文本上完全看不出来。
CAMPLUS_MODELS: dict[str, tuple[str, str]] = {
    "zh": (
        "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        "f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11",
    ),
    # 英文模型尚未核实哈希；用它必须显式传本地路径，避免"能下但没校验"。
    "en": ("3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx", ""),
}

MIN_EMBED_FRAMES = 10

# 声纹嵌入在一秒以下是**噪声**，不是"较弱但还能用的信号"。实机 127 秒录音上量过：长段
# （1.3–4.2 秒）与同一个人的质心相似度 0.82–0.92，而 0.5 秒的"哈喽。"只有 0.36，几个
# 短句彼此之间也只有 0.17–0.50——它们不携带身份信息，任何基于它们的判定都是掷骰子。
# 拿噪声开新身份还是**不可逆**的：那条噪声向量会成为质心，把后面所有真正的语音都推开，
# 于是一个真人被拆成 4 个说话人。所以短片段不许开新身份、也不更新质心。
RELIABLE_EMBED_SEC = 1.0
EMBED_FBANK_BINS = 80


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def ensure_speaker_model(
    lang: str = "zh",
    *,
    cache_dir: str | Path | None = None,
    url_template: str = SPEAKER_MODEL_URL,
    timeout: float = 120.0,
) -> Path:
    """返回本地声纹模型路径，必要时下载并校验 sha256。"""
    try:
        name, want_sha = CAMPLUS_MODELS[lang]
    except KeyError:
        raise ValueError(
            f"unknown speaker model language {lang!r}; known: {sorted(CAMPLUS_MODELS)}"
        ) from None

    root = Path(cache_dir).expanduser() if cache_dir else SPEAKER_MODEL_CACHE
    root.mkdir(parents=True, exist_ok=True)
    dest = root / name
    if dest.exists() and want_sha and _sha256(dest) == want_sha:
        return dest

    if not want_sha:
        raise RuntimeError(
            f"{name} has no pinned sha256 in this module, so it will not be downloaded; "
            "fetch it yourself and pass the path via model_path"
        )

    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url_template.format(name), timeout=timeout) as response:
        with tmp.open("wb") as handle:
            while True:
                block = response.read(1 << 16)
                if not block:
                    break
                handle.write(block)
    got = _sha256(tmp)
    if got != want_sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {name}: expected {want_sha}, got {got}")
    tmp.replace(dest)
    return dest


class OnnxCampplusEmbedder:
    """3D-Speaker CAM++ 声纹嵌入（ONNX，不依赖 torch）。

    链路：kaldi-native-fbank 提 80 维 fbank（25ms 窗 / 10ms 移，dither=0）→
    onnxruntime → 192 维嵌入，做 L2 归一化。

    预处理按模型自带元数据：``normalize_samples=1``（波形按峰值归一）与
    ``feature_normalize_type=global-mean``（**逐维**减均值，即 CMN）。三种归一
    实测都能分开说话人，CMN 的分离余量最宽（0.834 对 0.405），与元数据一致。

    依赖 ``onnxruntime`` 与 ``kaldi-native-fbank``，**刻意在本方法内 import**：
    realtime 包必须在本机没装这两个可选依赖时也能导入，否则"轻量导入不拖重依赖"
    这条在包这一层就破了。
    """

    embedding_dim = 192

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        lang: str = "zh",
        cache_dir: str | Path | None = None,
        providers: list[str] | None = None,
        sample_rate: int = 16000,
    ):
        import kaldi_native_fbank  # noqa: PLC0415 - 可选依赖，见类文档
        import onnxruntime  # noqa: PLC0415

        self._knf = kaldi_native_fbank
        self._sample_rate = int(sample_rate)
        self._path = (
            Path(model_path).expanduser() if model_path else ensure_speaker_model(lang, cache_dir=cache_dir)
        )
        self._session = onnxruntime.InferenceSession(
            str(self._path), providers=providers or ["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self.embedding_dim = int(self._session.get_outputs()[0].shape[-1])

    @property
    def model_path(self) -> Path:
        return self._path

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
        frames = self._fbank(audio)
        if frames.shape[0] < MIN_EMBED_FRAMES:
            return None
        vec = self._session.run(None, {self._input: frames[None, :, :]})[0]
        return _normalize(np.asarray(vec, dtype=np.float32).reshape(-1))

    def _fbank(self, audio: np.ndarray) -> np.ndarray:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 0.0:
            samples = samples / peak           # 模型元数据 normalize_samples = 1

        options = self._knf.FbankOptions()
        options.frame_opts.samp_freq = self._sample_rate
        options.frame_opts.dither = 0.0
        options.mel_opts.num_bins = EMBED_FBANK_BINS
        fbank = self._knf.OnlineFbank(options)
        fbank.accept_waveform(self._sample_rate, samples)
        fbank.input_finished()

        ready = fbank.num_frames_ready
        if ready <= 0:
            return np.zeros((0, EMBED_FBANK_BINS), dtype=np.float32)
        frames = np.stack([fbank.get_frame(i) for i in range(ready)]).astype(np.float32)
        # feature_normalize_type = global-mean -> 逐维减均值（CMN）
        return frames - frames.mean(axis=0, keepdims=True)


class SpeakerGallery:
    """把窗口内的局部说话人标签映射到全局一致的说话人编号。

    模型输出的 ``[S01]``/``[S02]`` 只在单个窗口内自洽。跨窗口的一致性是靠声纹
    嵌入的余弦相似度匹配做出来的，局部标签只是"同一窗口内这些段落属于同一个人"
    这个先验。
    """

    def __init__(
        self,
        embedder: SpeakerEmbedder | None,
        *,
        threshold: float = 0.55,
        min_segment_sec: float = 0.4,
        min_reliable_sec: float = RELIABLE_EMBED_SEC,
        sample_rate: int = 16000,
    ):
        self._embedder = embedder
        self._threshold = float(threshold)
        self._sample_rate = int(sample_rate)
        self._min_samples = max(1, int(round(min_segment_sec * self._sample_rate)))
        self._reliable_samples = max(1, int(round(min_reliable_sec * self._sample_rate)))
        self._entries: dict[str, _GalleryEntry] = {}
        self._next_index = 1
        self._unknown_assignments = 0

    @property
    def enabled(self) -> bool:
        return self._embedder is not None

    def assign(
        self,
        segments: list[Segment],
        audio_of: Callable[[Segment], np.ndarray | None],
    ) -> list[Assignment]:
        if not segments:
            return []
        if self._embedder is None:
            return [Assignment(seg, seg.speaker, False) for seg in segments]

        groups: dict[tuple[int, str], list[Segment]] = {}
        order: list[tuple[int, str]] = []
        for seg in segments:
            key = (seg.window_id, seg.speaker)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(seg)

        decided: dict[tuple[int, str], tuple[str, bool]] = {}
        for key in order:
            decided[key] = self._decide_group(groups[key], audio_of)

        return [
            Assignment(seg, *decided[(seg.window_id, seg.speaker)]) for seg in segments
        ]

    def _decide_group(
        self,
        group: list[Segment],
        audio_of: Callable[[Segment], np.ndarray | None],
    ) -> tuple[str, bool]:
        best_audio: np.ndarray | None = None
        best_size = 0
        for seg in group:
            audio = audio_of(seg)
            if audio is None:
                continue
            size = int(np.asarray(audio).size)
            if size < self._min_samples:
                continue
            if size > best_size:
                best_audio, best_size = audio, size

        if best_audio is None:
            self._unknown_assignments += 1
            return UNKNOWN_SPEAKER_ID, False

        embedding = self._embedder.embed(best_audio, self._sample_rate)
        if embedding is None:
            self._unknown_assignments += 1
            return UNKNOWN_SPEAKER_ID, False

        return self._match(embedding, samples=best_size)

    def _match(self, embedding: np.ndarray, *, samples: int) -> tuple[str, bool]:
        vec = _normalize(embedding)
        best_id: str | None = None
        best_score = -1.0
        for entry in self._entries.values():
            score = float(np.dot(entry.centroid, vec))
            if score > best_score:
                best_id, best_score = entry.id, score

        if best_id is not None and best_score >= self._threshold:
            entry = self._entries[best_id]
            entry.centroid = _normalize(entry.centroid * entry.samples + vec)
            entry.samples += 1
            return best_id, True

        if samples >= self._reliable_samples:
            new_id = f"S{self._next_index:02d}"
            self._next_index += 1
            self._entries[new_id] = _GalleryEntry(id=new_id, centroid=vec, samples=1)
            return new_id, True

        # 太短、又匹配不上：这段音频的向量是噪声，判定不可信（见 RELIABLE_EMBED_SEC）。
        # 有已有说话人就归到最近的那个，但**不更新质心**——噪声进质心会把后面真正属于
        # 那个人的语音全都推开，那是这个 bug 里最难恢复的一步。没有就报"未知"。
        if best_id is not None:
            return best_id, False
        self._unknown_assignments += 1
        return UNKNOWN_SPEAKER_ID, False

    def rename(self, speaker_id: str, name: str) -> None:
        entry = self._entries.get(speaker_id)
        if entry is None:
            raise KeyError(speaker_id)
        entry.name = str(name).strip()

    def display_name(self, speaker_id: str) -> str:
        entry = self._entries.get(speaker_id)
        if entry is None:
            return UNKNOWN_SPEAKER_NAME if speaker_id == UNKNOWN_SPEAKER_ID else speaker_id
        return entry.name or entry.id

    def speakers(self) -> list[dict]:
        roster = [
            {"id": entry.id, "name": entry.name or entry.id, "samples": entry.samples}
            for entry in self._entries.values()
        ]
        if self._unknown_assignments:
            roster.append(
                {"id": UNKNOWN_SPEAKER_ID, "name": UNKNOWN_SPEAKER_NAME, "samples": self._unknown_assignments}
            )
        return roster

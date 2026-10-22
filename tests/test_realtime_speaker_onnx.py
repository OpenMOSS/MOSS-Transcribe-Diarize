"""`OnnxCampplusEmbedder` 与模型获取。

需要真实模型文件的用例用 `@unittest.skipUnless` 跳过（模型在
`~/.cache/mtd-speaker/`，首次调用 `OnnxCampplusEmbedder()` 会自动下载并校验
sha256）。不用真模型的那三条永远跑，它们钉的是另外两件事：**可选依赖必须是真的
惰性**，以及**没有锁定哈希的模型拒绝下载**。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from moss_transcribe_diarize.realtime.speaker import (
    CAMPLUS_MODELS,
    MIN_EMBED_FRAMES,
    OnnxCampplusEmbedder,
    ensure_speaker_model,
)

MODEL_NAME, MODEL_SHA = CAMPLUS_MODELS["zh"]


def _model_available() -> bool:
    return (Path.home() / ".cache" / "mtd-speaker" / MODEL_NAME).exists()


class OptionalDependencyTest(unittest.TestCase):
    def test_speaker_module_imports_without_the_optional_dependencies(self):
        """onnxruntime / kaldi-native-fbank 必须只在真正取用嵌入器时才 import。

        它们若被放到模块顶层，realtime 包就要求每个使用它的人都装 onnxruntime——
        而 phase 1 那条 torch-free 约束要的正是"轻量导入不拖重依赖"。
        """
        code = (
            "import sys\n"
            "for name in ('onnxruntime', 'kaldi_native_fbank', 'torch', 'transformers'):\n"
            "    sys.modules[name] = None\n"
            "import moss_transcribe_diarize.realtime.speaker as sp\n"
            "assert hasattr(sp, 'OnnxCampplusEmbedder')\n"
            "assert hasattr(sp, 'SpeakerGallery')\n"
            "print('IMPORT_OK')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertIn("IMPORT_OK", result.stdout, result.stderr)

    def test_constructing_the_embedder_does_require_them(self):
        """反过来也要成立：没装依赖时构造应当报错，而不是悄悄用一个空实现。"""
        code = (
            "import sys\n"
            "for name in ('onnxruntime', 'kaldi_native_fbank'):\n"
            "    sys.modules[name] = None\n"
            "from moss_transcribe_diarize.realtime.speaker import OnnxCampplusEmbedder\n"
            "try:\n"
            "    OnnxCampplusEmbedder(model_path='does-not-matter.onnx')\n"
            "except ImportError:\n"
            "    print('IMPORT_ERROR_OK')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertIn("IMPORT_ERROR_OK", result.stdout, result.stderr)


class EnsureSpeakerModelTest(unittest.TestCase):
    def test_rejects_unknown_language(self):
        with self.assertRaises(ValueError):
            ensure_speaker_model("klingon")

    def test_refuses_a_model_without_a_pinned_sha256(self):
        """英文模型在本模块里没有核实过的哈希——那就**不要**下载。

        能下但不校验，等于把"声纹被换掉会静默破坏说话人一致性"这条风险留在代码里。
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                ensure_speaker_model("en", cache_dir=tmp)

    def test_english_entry_has_no_hash(self):
        self.assertEqual(CAMPLUS_MODELS["en"][1], "")
        self.assertNotEqual(CAMPLUS_MODELS["zh"][1], "")


@unittest.skipUnless(_model_available(), f"{MODEL_NAME} 未缓存；先跑一次 OnnxCampplusEmbedder() 会自动下载")
class OnnxCampplusEmbedderTest(unittest.TestCase):
    """以下需要真实模型（约 28 MB）。"""

    @classmethod
    def setUpClass(cls):
        cls.embedder = OnnxCampplusEmbedder()

    def test_embedding_dimension_matches_the_model(self):
        self.assertEqual(self.embedder.embedding_dim, 192)

    def test_embedding_is_unit_norm(self):
        import numpy as np

        audio = np.sin(2 * np.pi * 180.0 * np.arange(16000, dtype=np.float32) / 16000.0)
        vec = self.embedder.embed(audio, 16000)

        self.assertIsNotNone(vec)
        self.assertEqual(vec.shape, (192,))
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)

    def test_too_short_audio_returns_none(self):
        import numpy as np

        # 短于 MIN_EMBED_FRAMES 帧（10 帧 @10ms 移 ≈ 0.1 秒）就放弃，而不是硬算
        self.assertIsNone(self.embedder.embed(np.zeros(200, dtype=np.float32), 16000))

    def test_silence_does_not_crash(self):
        import numpy as np

        # 峰值归一会除以 0；这里钉住它被挡住了
        result = self.embedder.embed(np.zeros(16000, dtype=np.float32), 16000)
        self.assertTrue(result is None or result.shape == (192,))
        self.assertGreaterEqual(MIN_EMBED_FRAMES, 1)

    def test_cached_model_is_used_without_downloading(self):
        path = ensure_speaker_model("zh")
        self.assertTrue(path.exists())
        self.assertTrue(self.embedder.model_path.samefile(path))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import subprocess
import sys
import unittest


def _run_with_blocked(module: str, blocked: tuple[str, ...]) -> subprocess.CompletedProcess:
    """在子进程里导入 module，并让 blocked 里的顶层包不可用。"""
    code = (
        "import sys\n"
        f"for name in {blocked!r}:\n"
        "    sys.modules[name] = None\n"
        f"import {module}\n"
        "print('IMPORT_OK')\n"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


class LazyPackageImportTest(unittest.TestCase):
    def test_realtime_import_path_is_torch_free(self):
        """父包 ``__init__`` 必须在 torch/transformers 被阻断时也能导入。

        实时进程导入 ``moss_transcribe_diarize.realtime.*`` 时，Python 会先执行
        父包 ``__init__``；只要父包在导入期就需要 torch，整条实时路径就被拖累。
        改动前（RED）这里会在 ``configuration_moss_transcribe_diarize`` 处因
        ``transformers`` 被阻断而抛 ``ModuleNotFoundError``，所以本断言在 RED
        阶段确实失败——它区分得开改动前后。
        """
        result = _run_with_blocked("moss_transcribe_diarize", ("torch", "transformers"))
        self.assertIn("IMPORT_OK", result.stdout, result.stderr)
        self.assertNotIn("torch", result.stderr.lower())

    def test_prompts_module_imports_without_torch(self):
        result = _run_with_blocked("moss_transcribe_diarize.prompts", ("torch", "transformers"))
        self.assertIn("IMPORT_OK", result.stdout, result.stderr)
        self.assertNotIn("torch", result.stderr.lower())

    def test_torch_backed_module_import_fails_when_blocked(self):
        """正向对照：阻断确实生效——依赖 torch 的子模块在这种解释器里必败。

        没有这一条，上面两条的 ``IMPORT_OK`` 断言可能是空转（万一阻断手段失效，
        它们仍会通过）。这里断言失败原因来自被阻断的 torch 本身
        （``halted; None in sys.modules``），而不是别的偶然错误。
        """
        result = _run_with_blocked(
            "moss_transcribe_diarize.modeling_moss_transcribe_diarize",
            ("torch", "transformers"),
        )
        self.assertNotIn("IMPORT_OK", result.stdout)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("halted", result.stderr)
        self.assertIn("torch", result.stderr.lower())

    def test_lazy_attribute_still_resolves(self):
        """惰性解析确实经 ``__getattr__`` 发生，并缓存结果（PEP 562）。

        在全新解释器里运行：确保这些名字尚未被任何先前的访问缓存。导入子模块
        只把子模块挂到包上，不会引入它内部的这些名字，因此访问前的“不在
        ``vars(mtd)``”断言为真；此后通过包访问才触发 ``__getattr__``，得到正确
        对象并写入包 globals。既覆盖 torch-free 名（``DEFAULT_PROMPT``），也覆盖
        torch-backed 名（``MossTranscribeDiarizeModel``）。
        """
        result = _run(
            "import moss_transcribe_diarize as mtd\n"
            "assert 'DEFAULT_PROMPT' not in vars(mtd), 'DEFAULT_PROMPT cached early'\n"
            "assert 'MossTranscribeDiarizeModel' not in vars(mtd), 'model cached early'\n"
            "from moss_transcribe_diarize.prompts import DEFAULT_PROMPT\n"
            "from moss_transcribe_diarize.modeling_moss_transcribe_diarize import (\n"
            "    MossTranscribeDiarizeModel,\n"
            ")\n"
            "# 导入子模块不应把这些名字带进包命名空间\n"
            "assert 'DEFAULT_PROMPT' not in vars(mtd), 'cached by submodule import'\n"
            "assert 'MossTranscribeDiarizeModel' not in vars(mtd), 'cached by submodule import'\n"
            "# 通过包访问 -> 触发 __getattr__\n"
            "assert mtd.DEFAULT_PROMPT is DEFAULT_PROMPT\n"
            "assert mtd.MossTranscribeDiarizeModel is MossTranscribeDiarizeModel\n"
            "# 解析结果被缓存\n"
            "assert vars(mtd)['DEFAULT_PROMPT'] is DEFAULT_PROMPT\n"
            "assert vars(mtd)['MossTranscribeDiarizeModel'] is MossTranscribeDiarizeModel\n"
            "print('IMPORT_OK')\n"
        )
        self.assertIn("IMPORT_OK", result.stdout, result.stderr)

    def test_unknown_attribute_raises(self):
        import moss_transcribe_diarize as mtd

        with self.assertRaises(AttributeError):
            mtd.definitely_not_a_real_name

    def test_all_names_are_importable(self):
        import moss_transcribe_diarize as mtd

        for name in mtd.__all__:
            self.assertTrue(hasattr(mtd, name), name)


if __name__ == "__main__":
    unittest.main()

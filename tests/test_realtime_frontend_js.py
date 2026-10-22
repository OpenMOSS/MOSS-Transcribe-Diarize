"""把实时页面纯函数的 Node 测试接进 pytest。

前端没有构建步骤、也没有测试运行器，所以"能被纯函数表达的判定"都抽到
``static/realtime_logic.js``，由 ``tests/js/`` 用 ``node --test`` 跑，这里只是把它
挂进同一个套件——否则它就是一条没人跑的测试。没装 node 时跳过，不让整套测试挂掉。
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

NODE = shutil.which("node")
TEST_FILE = Path(__file__).resolve().parent / "js" / "realtime_logic.test.mjs"


@unittest.skipUnless(NODE, "node is not installed")
class RealtimeFrontendLogicTest(unittest.TestCase):
    def test_the_pure_logic_passes(self):
        result = subprocess.run(
            [NODE, "--test", str(TEST_FILE)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("fail 0", result.stdout)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import unittest
from unittest.mock import patch

from moss_transcribe_diarize.app.server import ERROR_STATUS_CODES, STATIC_DIR, create_app


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")


class LocaleCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        locale_dir = STATIC_DIR / "locales"
        cls.catalogs = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(locale_dir.glob("*.json"))
        }

    def test_expected_locales_have_matching_keys_and_placeholders(self):
        self.assertEqual(set(self.catalogs), {"en", "zh-CN"})
        english = self.catalogs["en"]
        chinese = self.catalogs["zh-CN"]
        self.assertEqual(set(english), set(chinese))
        for key in english:
            with self.subTest(key=key):
                self.assertEqual(
                    set(PLACEHOLDER_RE.findall(english[key])),
                    set(PLACEHOLDER_RE.findall(chinese[key])),
                )

    def test_static_and_dynamic_translation_keys_exist(self):
        english = self.catalogs["en"]
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        html_keys = set(re.findall(r'data-i18n(?:-[a-z-]+)?="([^"]+)"', html))
        js_keys = set(re.findall(r"\bt\('([^']+)'", app_js))
        plural_keys = set(re.findall(r"\btp\('([^']+)'", app_js))
        fallback_keys = set(re.findall(r"localizedError\([^)]*, '([^']+)'\)", app_js))
        self.assertFalse((html_keys | js_keys | fallback_keys) - set(english))
        for key in plural_keys:
            self.assertIn(f"{key}.one", english)
            self.assertIn(f"{key}.other", english)

    def test_all_known_api_error_codes_are_translated(self):
        keys = set(self.catalogs["en"])
        for code in {*ERROR_STATUS_CODES, "invalid_segments"}:
            self.assertIn(f"errors.{code}", keys)

    def test_realtime_frontend_uses_only_known_keys(self):
        """实时页面用到的每个词条都必须在 en 里存在（zh-CN 由上面那条保证一致）。"""
        english = self.catalogs["en"]
        html = (STATIC_DIR / "realtime.html").read_text(encoding="utf-8")
        script = "".join(
            (STATIC_DIR / name).read_text(encoding="utf-8")
            for name in ("realtime.js", "realtime_logic.js")
        )
        html_keys = set(re.findall(r'data-i18n(?:-[a-z-]+)?="([^"]+)"', html))
        # 引号与边界写成 chr()/字符类，是为了这条正则里不出现任何转义序列。
        quotes = '\'"'
        # 直接抓带命名空间的字符串字面量，而不是抓 `t(...)` 的调用形状：词条也会经
        # `notice(...)` 这类包装传进去，只认一种调用形状会漏掉一半。
        script_keys = set(re.findall(
            rf"[{quotes}]((?:realtime|errors|runtime)[.][A-Za-z0-9_.]+)[{quotes}]", script,
        ))
        fallback_keys = set(re.findall(
            rf"localizedError[(][^)]*, [{quotes}]([^{quotes}]+)[{quotes}][)]", script,
        ))
        missing = (html_keys | script_keys | fallback_keys) - set(english)
        self.assertFalse(missing, f"missing keys: {sorted(missing)}")

    def test_the_realtime_namespace_exists_in_both_locales(self):
        for locale, catalog in self.catalogs.items():
            with self.subTest(locale=locale):
                self.assertTrue([k for k in catalog if k.startswith("realtime.")])
        realtime_en = {k for k in self.catalogs["en"] if k.startswith("realtime.")}
        realtime_zh = {k for k in self.catalogs["zh-CN"] if k.startswith("realtime.")}
        self.assertEqual(realtime_en, realtime_zh)

    def test_the_realtime_frontend_is_served_and_references_real_assets(self):
        html = (STATIC_DIR / "realtime.html").read_text(encoding="utf-8")
        for asset in ("assets/realtime.js", "assets/realtime.css"):
            self.assertIn(asset, html)
            self.assertTrue((STATIC_DIR / asset.split("/", 1)[1]).is_file(), asset)
        self.assertTrue((STATIC_DIR / "audio-worklet.js").is_file())


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class AppI18nApiTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = create_app(model_path="fake-model", runs_dir=self.temp_dir.name)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_frontend_static_assets_are_served(self):
        expected = {
            "/": "text/html",
            "/favicon.svg": "image/svg+xml",
            "/assets/styles.css": "text/css",
            "/assets/app.js": "text/javascript",
            "/assets/i18n.js": "text/javascript",
            "/assets/locales/zh-CN.json": "application/json",
            "/assets/locales/en.json": "application/json",
        }
        for url, media_type in expected.items():
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assertIn(media_type, response.headers["content-type"])

    def test_job_and_file_errors_include_stable_codes(self):
        missing = self.client.get("/api/jobs/not-found")
        self.assert_error(missing, 404, "job_not_found")

        job, _ = self.app.state.manager.create_job_for_upload("sample.wav")
        self.assert_error(self.client.delete(f"/api/jobs/{job.id}"), 409, "job_running")
        self.assert_error(self.client.get(f"/api/jobs/{job.id}/media"), 404, "media_missing")
        self.assert_error(
            self.client.get(f"/api/jobs/{job.id}/download?kind=srt"),
            404,
            "file_not_ready",
        )

    def test_validation_errors_include_stable_codes(self):
        cases = [
            ({"decoding": "invalid"}, "invalid_decoding"),
            ({"max_len": "0"}, "invalid_max_length"),
            ({"max_new_tokens": "0"}, "invalid_max_new_tokens"),
            ({"decoding": "sample", "temperature": "0"}, "invalid_temperature"),
        ]
        for payload, code in cases:
            with self.subTest(code=code):
                response = self.client.post(
                    "/api/jobs",
                    files={"file": ("sample.wav", b"audio", "audio/wav")},
                    data=payload,
                )
                self.assert_error(response, 400, code)

        job, _ = self.app.state.manager.create_job_for_upload("sample.wav")
        invalid_segments = self.client.put(f"/api/jobs/{job.id}/segments", json={})
        self.assert_error(invalid_segments, 400, "invalid_segments")

    def test_render_preconditions_include_stable_codes(self):
        job, _ = self.app.state.manager.create_job_for_upload("sample.wav")

        class Missing:
            available = False

        with patch("moss_transcribe_diarize.app.jobs.detect_ffmpeg", return_value=Missing()):
            unavailable = self.client.post(f"/api/jobs/{job.id}/render", json={"style": {}})
        self.assert_error(unavailable, 503, "ffmpeg_unavailable")

        class Available:
            available = True

        with patch("moss_transcribe_diarize.app.jobs.detect_ffmpeg", return_value=Available()):
            no_subtitles = self.client.post(f"/api/jobs/{job.id}/render", json={"style": {}})
        self.assert_error(no_subtitles, 503, "subtitles_unavailable")

    def assert_error(self, response, status: int, code: str):
        self.assertEqual(response.status_code, status)
        payload = response.json()
        self.assertEqual(payload["code"], code)
        self.assertIsInstance(payload["detail"], str)
        self.assertTrue(payload["detail"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import threading
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from moss_transcribe_diarize.app.model_runner import TranscriptionResult


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


class FakeRunner:
    model_path = "fake-model"

    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs.get("status_callback")
        if callback:
            callback("transcribing", 0.5)
        return TranscriptionResult(
            text="[0][S01]hello[1.5]",
            prompt_len=10,
            generated_tokens=5,
            elapsed_sec=0.01,
            model="fake-model",
            audio=str(audio_path),
            decoding="greedy",
            temperature=None,
        )


class BlockingRunner:
    model_path = "fake-model"

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def transcribe(self, audio_path, **kwargs):
        callback = kwargs.get("status_callback")
        if callback:
            callback("transcribing", 0.55, 3)
        self.started.set()
        self.release.wait(timeout=2)
        return TranscriptionResult(
            text="[0][S01]hello[1.5]",
            prompt_len=10,
            generated_tokens=5,
            elapsed_sec=0.01,
            model="fake-model",
            audio=str(audio_path),
            decoding="greedy",
            temperature=None,
        )


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class AppApiTest(unittest.TestCase):
    def test_runtime_reports_vllm_backend(self):
        from fastapi.testclient import TestClient
        from moss_transcribe_diarize.app.server import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(
                model_path="unused-local-model",
                runs_dir=tmpdir,
                backend="vllm",
                vllm_base_url="http://vllm.test:8000/v1",
                vllm_model="moss-served",
            )
            client = TestClient(app)
            runtime = client.get("/api/runtime")
            self.assertEqual(runtime.status_code, 200)
            model = runtime.json()["model"]
            self.assertEqual(model["backend"], "vllm")
            self.assertEqual(model["path"], "moss-served")
            self.assertEqual(model["base_url"], "http://vllm.test:8000/v1")

    def test_job_lifecycle_and_missing_ffmpeg_render_error(self):
        from fastapi.testclient import TestClient
        from moss_transcribe_diarize.app.server import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            runner = FakeRunner()
            app.state.manager.model_runner = runner
            client = TestClient(app)

            created = client.post(
                "/api/jobs",
                files={"file": ("sample.wav", b"audio", "audio/wav")},
                data={
                    "prompt": "custom prompt",
                    "max_new_tokens": "5",
                    "max_len": "456",
                    "decoding": "sample",
                    "temperature": "0.7",
                },
            )
            self.assertEqual(created.status_code, 200)
            job_id = created.json()["id"]

            job = {}
            for _ in range(40):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] == "waiting_review":
                    break
                time.sleep(0.05)
            self.assertEqual(job["status"], "waiting_review")
            self.assertEqual(job["inference"]["prompt"], "custom prompt")
            self.assertEqual(job["inference"]["max_new_tokens"], 5)
            self.assertEqual(job["inference"]["max_length"], 456)
            self.assertEqual(job["inference"]["decoding"], "sample")
            self.assertEqual(job["inference"]["temperature"], 0.7)
            self.assertEqual(job["usage"]["generated_tokens"], 5)
            self.assertEqual(job["usage"]["max_new_tokens"], 5)
            self.assertTrue(job["usage"]["possibly_truncated"])
            self.assertEqual(runner.calls[-1]["prompt"], "custom prompt")
            self.assertEqual(runner.calls[-1]["max_new_tokens"], 5)
            self.assertEqual(runner.calls[-1]["max_length"], 456)
            self.assertEqual(runner.calls[-1]["decoding"], "sample")
            self.assertEqual(runner.calls[-1]["temperature"], 0.7)

            listed = client.get("/api/jobs")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(listed.json()["jobs"][0]["id"], job_id)

            media = client.get(f"/api/jobs/{job_id}/media")
            self.assertEqual(media.status_code, 200)

            rerun = client.post(f"/api/jobs/{job_id}/rerun", json={"max_new_tokens": 10})
            self.assertEqual(rerun.status_code, 200)
            rerun_id = rerun.json()["id"]
            self.assertNotEqual(rerun_id, job_id)
            rerun_job = {}
            for _ in range(40):
                rerun_job = client.get(f"/api/jobs/{rerun_id}").json()
                if rerun_job["status"] == "waiting_review":
                    break
                time.sleep(0.05)
            self.assertEqual(rerun_job["status"], "waiting_review")
            self.assertEqual(rerun_job["media_name"], "sample.wav")
            self.assertEqual(rerun_job["inference"]["max_new_tokens"], 10)
            self.assertEqual(runner.calls[-1]["max_new_tokens"], 10)

            segments = client.get(f"/api/jobs/{job_id}/segments").json()["segments"]
            self.assertEqual(segments[0]["speaker"], "S01")
            segments[0]["text"] = "edited"
            updated = client.put(
                f"/api/jobs/{job_id}/segments",
                json={"segments": segments, "style": {"speaker_names": {"S01": "Alice"}}},
            )
            self.assertEqual(updated.status_code, 200)
            self.assertEqual(updated.json()["segments"][0]["text"], "edited")
            self.assertEqual(client.get(f"/api/jobs/{job_id}").json()["subtitle_style"]["speaker_names"]["S01"], "Alice")

            download = client.get(f"/api/jobs/{job_id}/download?kind=srt")
            self.assertEqual(download.status_code, 200)
            self.assertIn("edited", download.text)
            self.assertIn("Alice: edited", download.text)

            class Missing:
                available = False

            with patch("moss_transcribe_diarize.app.jobs.detect_ffmpeg", return_value=Missing()):
                render = client.post(f"/api/jobs/{job_id}/render", json={"style": {}})
            self.assertEqual(render.status_code, 503)

            deleted = client.delete(f"/api/jobs/{job_id}")
            self.assertEqual(deleted.status_code, 200)
            self.assertEqual(client.get(f"/api/jobs/{job_id}").status_code, 404)

    def test_running_job_exposes_live_token_progress(self):
        from fastapi.testclient import TestClient
        from moss_transcribe_diarize.app.server import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            runner = BlockingRunner()
            app.state.manager.model_runner = runner
            client = TestClient(app)

            created = client.post(
                "/api/jobs",
                files={"file": ("sample.wav", b"audio", "audio/wav")},
                data={"max_new_tokens": "5"},
            )
            self.assertEqual(created.status_code, 200)
            job_id = created.json()["id"]
            self.assertTrue(runner.started.wait(timeout=2))

            running = client.get(f"/api/jobs/{job_id}").json()
            self.assertEqual(running["status"], "transcribing")
            self.assertEqual(running["usage"]["generated_tokens"], 3)
            self.assertEqual(running["usage"]["max_new_tokens"], 5)
            self.assertAlmostEqual(running["progress"], 0.55)

            runner.release.set()
            finished = {}
            for _ in range(40):
                finished = client.get(f"/api/jobs/{job_id}").json()
                if finished["status"] == "waiting_review":
                    break
                time.sleep(0.05)
            self.assertEqual(finished["status"], "waiting_review")
            self.assertEqual(finished["usage"]["generated_tokens"], 5)

    def test_rendering_job_rejects_subtitle_updates(self):
        from fastapi.testclient import TestClient
        from moss_transcribe_diarize.app.server import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            app.state.manager.model_runner = FakeRunner()
            client = TestClient(app)

            created = client.post(
                "/api/jobs",
                files={"file": ("sample.wav", b"audio", "audio/wav")},
            )
            job_id = created.json()["id"]
            for _ in range(40):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] == "waiting_review":
                    break
                time.sleep(0.05)
            self.assertEqual(job["status"], "waiting_review")

            original = client.get(f"/api/jobs/{job_id}/segments").json()["segments"]
            edited = [dict(item) for item in original]
            edited[0]["text"] = "must not be saved"
            app.state.manager._set_status(
                app.state.manager.get_job(job_id), "rendering", 0.97, error=None
            )

            updated = client.put(
                f"/api/jobs/{job_id}/segments",
                json={"segments": edited},
            )

            self.assertEqual(updated.status_code, 409)
            self.assertIn("while the job is rendering", updated.json()["detail"])
            self.assertEqual(
                client.get(f"/api/jobs/{job_id}/segments").json()["segments"],
                original,
            )
            self.assertIn(
                "const EDIT_STATES = new Set(['waiting_review', 'done']);",
                client.get("/").text,
            )

    def test_duplicate_render_request_is_rejected_before_starting_worker(self):
        from fastapi.testclient import TestClient
        from moss_transcribe_diarize.app.server import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            app.state.manager.model_runner = FakeRunner()
            client = TestClient(app)

            created = client.post(
                "/api/jobs",
                files={"file": ("sample.wav", b"audio", "audio/wav")},
            )
            job_id = created.json()["id"]
            for _ in range(40):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] == "waiting_review":
                    break
                time.sleep(0.05)
            self.assertEqual(job["status"], "waiting_review")

            class Available:
                available = True

            render_started = threading.Event()
            release_render = threading.Event()

            def block_render(*_args):
                render_started.set()
                release_render.wait(timeout=2)

            with (
                patch("moss_transcribe_diarize.app.jobs.detect_ffmpeg", return_value=Available()),
                patch.object(app.state.manager, "_render_job", side_effect=block_render) as render_job,
            ):
                first = client.post(f"/api/jobs/{job_id}/render", json={"style": {}})
                self.assertEqual(first.status_code, 200)
                self.assertEqual(first.json()["status"], "rendering")
                self.assertTrue(render_started.wait(timeout=2))

                duplicate = client.post(f"/api/jobs/{job_id}/render", json={"style": {}})

                self.assertEqual(duplicate.status_code, 409)
                self.assertIn("already rendering", duplicate.json()["detail"])
                self.assertEqual(render_job.call_count, 1)
                release_render.set()

    def test_restart_recovers_interrupted_render_for_review(self):
        from fastapi.testclient import TestClient
        from moss_transcribe_diarize.app.server import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            app.state.manager.model_runner = FakeRunner()
            client = TestClient(app)

            job_ids = []
            for name in ("rendering.wav", "transcribing.wav"):
                created = client.post(
                    "/api/jobs",
                    files={"file": (name, b"audio", "audio/wav")},
                )
                job_ids.append(created.json()["id"])
            for job_id in job_ids:
                for _ in range(40):
                    job = client.get(f"/api/jobs/{job_id}").json()
                    if job["status"] == "waiting_review":
                        break
                    time.sleep(0.05)
                self.assertEqual(job["status"], "waiting_review")

            rendering_id, transcribing_id = job_ids
            app.state.manager._set_status(
                app.state.manager.get_job(rendering_id), "rendering", 0.97, error=None
            )
            app.state.manager._set_status(
                app.state.manager.get_job(transcribing_id), "transcribing", 0.5, error=None
            )

            restarted = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            restarted_client = TestClient(restarted)
            recovered = restarted_client.get(f"/api/jobs/{rendering_id}").json()

            self.assertEqual(recovered["status"], "waiting_review")
            self.assertEqual(recovered["progress"], 0.95)
            self.assertIn("interrupted", recovered["error"])
            self.assertIn("retry rendering", recovered["error"])
            segments = restarted_client.get(f"/api/jobs/{rendering_id}/segments").json()["segments"]
            edited = [dict(item) for item in segments]
            edited[0]["text"] = "editable after restart"
            updated = restarted_client.put(
                f"/api/jobs/{rendering_id}/segments",
                json={"segments": edited},
            )
            self.assertEqual(updated.status_code, 200)

            failed = restarted_client.get(f"/api/jobs/{transcribing_id}").json()
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["progress"], 1.0)
            self.assertEqual(failed["error"], "Interrupted by previous server shutdown.")


if __name__ == "__main__":
    unittest.main()

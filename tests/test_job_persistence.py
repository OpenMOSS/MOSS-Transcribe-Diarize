from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from moss_transcribe_diarize.app.jobs import JobManager, JobRecord


class FakeRunner:
    model_path = "fake-model"


class JobPersistenceTest(unittest.TestCase):
    def _make_job(self, runs_dir: Path) -> JobRecord:
        job_dir = runs_dir / "test-job"
        job_dir.mkdir()
        return JobRecord(
            id="test-job",
            status="waiting_review",
            media_name="interview.wav",
            input_path=str(job_dir / "input.wav"),
            job_dir=str(job_dir),
            inference_prompt="transcribe",
            max_length=4096,
            max_new_tokens=1024,
            decoding="greedy",
            temperature=None,
        )

    def test_interrupted_temp_write_preserves_last_valid_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            manager = JobManager.__new__(JobManager)
            job = self._make_job(runs_dir)
            manager._save_job(job)
            previous_payload = job.job_path.read_text(encoding="utf-8")

            job.status = "done"
            original_write_text = Path.write_text

            def interrupted_write(path, data, *args, **kwargs):
                original_write_text(path, data[:24], *args, **kwargs)
                raise OSError("simulated interrupted write")

            with patch.object(Path, "write_text", new=interrupted_write):
                with self.assertRaisesRegex(OSError, "simulated interrupted write"):
                    manager._save_job(job)

            self.assertEqual(job.job_path.read_text(encoding="utf-8"), previous_payload)
            self.assertEqual(json.loads(previous_payload)["status"], "waiting_review")
            self.assertEqual(list(job.job_path.parent.glob(".job.json.*.tmp")), [])

    def test_save_atomically_replaces_from_the_job_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            manager = JobManager.__new__(JobManager)
            job = self._make_job(runs_dir)

            with patch("moss_transcribe_diarize.app.jobs.os.replace", wraps=os.replace) as replace:
                manager._save_job(job)

            source, destination = replace.call_args.args
            self.assertEqual(Path(source).parent, job.job_path.parent)
            self.assertEqual(Path(destination), job.job_path)
            self.assertEqual(json.loads(job.job_path.read_text(encoding="utf-8"))["id"], job.id)
            self.assertEqual(list(job.job_path.parent.glob(".job.json.*.tmp")), [])

    def test_restart_ignores_orphan_temp_file_and_loads_last_valid_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            writer = JobManager.__new__(JobManager)
            job = self._make_job(runs_dir)
            writer._save_job(job)
            job.job_path.with_name(".job.json.orphan.tmp").write_text("{incomplete", encoding="utf-8")

            manager = JobManager(
                runs_dir,
                FakeRunner(),
                prompt="transcribe",
                max_length=4096,
                max_new_tokens=1024,
            )

            self.assertEqual(manager.get_job(job.id).status, "waiting_review")


if __name__ == "__main__":
    unittest.main()

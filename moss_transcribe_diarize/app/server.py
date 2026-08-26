import json
from pathlib import Path
from typing import Any

from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

from .ffmpeg import detect_ffmpeg
from .jobs import JobManager, JobManagerError
from .model_runner import ModelRunner
from .vllm_runner import VllmRunner


STATIC_DIR = Path(__file__).with_name("static")
INDEX_PATH = STATIC_DIR / "index.html"
ERROR_STATUS_CODES = {
    "job_not_found": 404,
    "job_running": 409,
    "media_missing": 404,
    "invalid_decoding": 400,
    "invalid_max_length": 400,
    "invalid_max_new_tokens": 400,
    "invalid_temperature": 400,
    "ffmpeg_unavailable": 503,
    "subtitles_unavailable": 503,
    "file_not_ready": 404,
}


def create_app(
    *,
    model_path: str | Path,
    runs_dir: str | Path = "runs",
    device: str = "auto",
    dtype: str = "bf16",
    prompt: str = DEFAULT_PROMPT,
    max_length: int = 131072,
    max_new_tokens: int = 2048,
    decoding: str = "greedy",
    temperature: float | None = None,
    backend: str = "hf",
    vllm_base_url: str | None = None,
    vllm_model: str | None = None,
    vllm_api_key: str | None = None,
    vllm_timeout: float = 600.0,
):
    try:
        from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError("Install fastapi, uvicorn, and python-multipart to run the local web app.") from exc

    app = FastAPI(title="MOSS Subtitle Studio")
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
    if backend == "vllm":
        if not vllm_base_url:
            raise ValueError("--vllm-base-url is required when backend='vllm'.")
        runner = VllmRunner(
            base_url=vllm_base_url,
            model=vllm_model or str(model_path),
            api_key=vllm_api_key,
            timeout=vllm_timeout,
        )
    else:
        runner = ModelRunner(model_path, device=device, dtype=dtype)
    manager = JobManager(
        runs_dir,
        runner,
        prompt=prompt,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        decoding=decoding,
        temperature=temperature,
    )
    app.state.manager = manager

    def error_response(code: str, detail: str, status_code: int):
        return JSONResponse({"detail": detail, "code": code}, status_code=status_code)

    def manager_error_response(exc: JobManagerError):
        return error_response(exc.code, str(exc), ERROR_STATUS_CODES.get(exc.code, 400))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.get("/favicon.svg")
    def favicon():
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml", headers={"Cache-Control": "no-store"})

    @app.get("/api/runtime")
    def runtime():
        return {
            "ffmpeg": detect_ffmpeg().to_dict(),
            "model": _runner_runtime_info(manager.model_runner),
            "inference": {
                "prompt": manager.prompt,
                "max_length": manager.max_length,
                "max_new_tokens": manager.max_new_tokens,
                "decoding": manager.decoding,
                "temperature": manager.temperature,
            },
        }

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [job.to_dict() for job in manager.list_jobs()]}

    @app.post("/api/jobs")
    async def create_job(
        file: UploadFile = File(...),
        prompt: str | None = Form(None),
        max_new_tokens: int | None = Form(None),
        max_len: int | None = Form(None),
        decoding: str | None = Form(None),
        temperature: float | None = Form(None),
    ):
        try:
            job, input_path = manager.create_job_for_upload(
                file.filename or "input.media",
                prompt=prompt,
                max_length=max_len,
                max_new_tokens=max_new_tokens,
                decoding=decoding,
                temperature=temperature,
            )
        except JobManagerError as exc:
            return manager_error_response(exc)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            with input_path.open("wb") as handle:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            manager.enqueue(job.id)
            return job.to_dict()
        except Exception as exc:
            manager._set_status(job, "failed", 1.0, error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return manager.get_job(job_id).to_dict()
        except JobManagerError as exc:
            return manager_error_response(exc)

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        try:
            manager.delete_job(job_id)
            return {"ok": True}
        except JobManagerError as exc:
            return manager_error_response(exc)

    @app.post("/api/jobs/{job_id}/rerun")
    async def rerun_job(job_id: str, request: Request):
        try:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            payload = payload if isinstance(payload, dict) else {}
            job = manager.rerun_job(
                job_id,
                prompt=payload.get("prompt"),
                max_length=payload["max_len"] if "max_len" in payload else payload.get("max_length"),
                max_new_tokens=payload.get("max_new_tokens"),
                decoding=payload.get("decoding"),
                temperature=payload.get("temperature"),
            )
            return job.to_dict()
        except JobManagerError as exc:
            return manager_error_response(exc)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/media")
    def media(job_id: str):
        try:
            job = manager.get_job(job_id)
            path = Path(job.input_path)
            if not path.exists():
                raise FileNotFoundError(str(path))
        except JobManagerError as exc:
            return manager_error_response(exc)
        except FileNotFoundError:
            return error_response("media_missing", "Media file is missing.", 404)
        return FileResponse(path, filename=path.name)

    @app.get("/api/jobs/{job_id}/segments")
    def get_segments(job_id: str):
        try:
            return {"segments": manager.list_segments(job_id)}
        except JobManagerError as exc:
            return manager_error_response(exc)

    @app.put("/api/jobs/{job_id}/segments")
    async def update_segments(job_id: str, request: Request):
        try:
            payload: Any = await request.json()
            segments = payload.get("segments", payload) if isinstance(payload, dict) else payload
            style = payload.get("style") if isinstance(payload, dict) else None
            if not isinstance(segments, list):
                return error_response(
                    "invalid_segments",
                    "Expected a JSON list or an object with a segments list.",
                    400,
                )
            return {"segments": manager.update_segments(job_id, segments, style)}
        except JobManagerError as exc:
            return manager_error_response(exc)
        except (KeyError, TypeError, ValueError) as exc:
            return error_response("invalid_segments", str(exc), 400)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/render")
    async def render(job_id: str, request: Request):
        try:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            job = manager.render(job_id, payload.get("style") if isinstance(payload, dict) else None)
            return job.to_dict()
        except JobManagerError as exc:
            return manager_error_response(exc)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str, kind: str):
        try:
            path = manager.download_path(job_id, kind)
        except JobManagerError as exc:
            return manager_error_response(exc)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, filename=path.name)

    return app


def _read_processor_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path).expanduser() / "processor_config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    keys = [
        "audio_tokens_per_second",
        "audio_merge_size",
        "time_marker_every_seconds",
        "enable_time_marker",
    ]
    return {key: data[key] for key in keys if key in data}


def _runner_runtime_info(runner) -> dict[str, Any]:
    if hasattr(runner, "runtime_info"):
        info = dict(runner.runtime_info())
    else:
        info = {
            "backend": "hf",
            "path": runner.model_path,
            "device": runner.device_name,
            "dtype": runner.dtype_name,
        }
    if info.get("backend") == "hf":
        info["processor"] = _read_processor_config(info.get("path") or "")
    else:
        info.setdefault("processor", {})
    return info

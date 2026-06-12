"""FastAPI application for queued FTV smoothing jobs."""

from __future__ import annotations

import argparse
import asyncio
import importlib.resources
import json
import logging
import mimetypes
import shutil
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

from .app_core import (
    CACHE_SCHEMA_VERSION,
    KEY_RE,
    TERMINAL_STATUSES,
    FTVAppConfig,
    has_completed_files,
    load_app_config,
    normalize_request,
    prepare_job_request,
    public_field_specs,
    read_result_metadata,
    result_dir,
)
from .backend import backend_diagnostics
from .job_store import JobStore
from .worker import celery_app, run_ftv_job

LOGGER = logging.getLogger(__name__)


def _asset_response(relative: str) -> Response:
    if "/" in relative or "\\" in relative or relative.startswith("."):
        raise HTTPException(status_code=404, detail="Asset not found")
    resource = importlib.resources.files("ftv_smoothing").joinpath("app_assets", relative)
    if not resource.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    media_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
    return Response(
        content=resource.read_bytes(),
        media_type=media_type,
        headers={"Cache-Control": "no-store"},
    )


def _resolve_result_path(results_dir: Path, cache_key: str, relative: str) -> Path:
    if not KEY_RE.fullmatch(cache_key):
        raise HTTPException(status_code=404, detail="Result not found")
    if not relative:
        raise HTTPException(status_code=404, detail="Result not found")
    base = result_dir(results_dir, cache_key).resolve()
    target = (base / relative).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Result not found") from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Result not found")
    return target


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _storage_summary(results_dir: Path, store: JobStore) -> dict[str, Any]:
    results_dir = results_dir.resolve()
    result_dirs = []
    referenced = store.referenced_cache_keys()
    if results_dir.exists():
        for item in results_dir.iterdir():
            if item.is_dir() and KEY_RE.fullmatch(item.name):
                result_dirs.append(
                    {
                        "cache_key": item.name,
                        "size_bytes": _directory_size(item),
                        "referenced": item.name in referenced,
                    }
                )
    result_dirs.sort(key=lambda item: item["size_bytes"], reverse=True)
    return {
        "results_dir": str(results_dir),
        "total_size_bytes": _directory_size(results_dir),
        "result_dir_count": len(result_dirs),
        "result_dirs": result_dirs[:50],
    }


def _revoke_celery_task(task_id: str | None) -> None:
    if not task_id:
        return
    control = getattr(celery_app, "control", None)
    if control is None:
        return
    try:
        control.revoke(task_id, terminate=True, signal="SIGTERM")
    except Exception:
        LOGGER.exception("Failed to revoke Celery task %s", task_id)


def build_app(
    config: FTVAppConfig | None = None,
    store: JobStore | None = None,
) -> FastAPI:
    """Build and configure the FastAPI app."""

    app_config = config or load_app_config()
    app_config.results_dir.mkdir(parents=True, exist_ok=True)
    job_store = store or JobStore(app_config.resolved_jobs_db, app_config.results_dir)
    app = FastAPI(title="FTV Calculation Console")

    @app.get("/")
    def index() -> Response:
        return _asset_response("index.html")

    @app.get("/static/{relative}")
    def static_asset(relative: str) -> Response:
        return _asset_response(relative)

    @app.get("/api/config")
    def api_config() -> dict[str, Any]:
        defaults = normalize_request({}, app_config.default_input_nc)
        input_path = Path(defaults["input_nc"]).expanduser()
        return {
            "defaults": defaults,
            "input_exists": input_path.exists(),
            "input_resolved": str(input_path.resolve()) if input_path.exists() else str(input_path),
            "results_dir": str(app_config.results_dir.resolve()),
            "jobs_db": str(app_config.resolved_jobs_db),
            "queue": "celery",
            "queues": {"cpu": "cpu", "gpu": "gpu"},
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "field_specs": public_field_specs(),
            "compute": backend_diagnostics(defaults["backend"]),
        }

    @app.get("/api/compute")
    def api_compute(requested: str = "auto") -> dict[str, Any]:
        try:
            return backend_diagnostics(requested)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs")
    def list_jobs() -> dict[str, Any]:
        return {"jobs": [job_store.public_job(job) for job in job_store.list_jobs()]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = job_store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job_store.public_job(job)

    def queue_for_params(params: dict[str, Any]) -> str:
        compute = backend_diagnostics(params["backend"])
        if params["backend"] == "gpu" and not compute["selected_gpu"]:
            raise HTTPException(status_code=400, detail=compute["reason"])
        return "gpu" if compute["selected_gpu"] else "cpu"

    def enqueue_job(job: dict[str, Any]) -> dict[str, Any]:
        queue = queue_for_params(job["parameters"])
        try:
            task = run_ftv_job.apply_async(args=[job["id"]], queue=queue)
        except Exception as exc:
            job_store.fail_job(job["id"], str(exc))
            raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}") from exc
        job_store.set_celery_task_id(job["id"], task.id)
        queued = job_store.get_job(job["id"])
        assert queued is not None
        return job_store.public_job(queued)

    def submit_payload(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            params, fingerprint, cache_key = prepare_job_request(
                payload,
                app_config.default_input_nc,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        cached = read_result_metadata(app_config.results_dir, cache_key)
        if (
            cached
            and cached.get("status") == "completed"
            and has_completed_files(app_config.results_dir, cached)
        ):
            job = job_store.create_job(
                cache_key=cache_key,
                parameters=params,
                input_fingerprint=fingerprint,
                status="completed",
                progress_percent=100,
                stage="Loaded from saved result",
                result_metadata={**cached, "cache_hit": True},
            )
            return job_store.public_job(job)

        active = job_store.find_active_by_cache_key(cache_key)
        if active is not None:
            return job_store.public_job(active)

        job = job_store.create_job(
            cache_key=cache_key,
            parameters=params,
            input_fingerprint=fingerprint,
            stage=f"Queued on {queue_for_params(params).upper()} worker",
        )
        return enqueue_job(job)

    @app.post("/api/jobs", status_code=202)
    def submit_job(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        return submit_payload(payload)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        job = job_store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] not in TERMINAL_STATUSES:
            _revoke_celery_task(job.get("celery_task_id"))
            job = job_store.cancel_job(job_id) or job
        return job_store.public_job(job)

    @app.post("/api/jobs/{job_id}/retry", status_code=202)
    def retry_job(job_id: str) -> dict[str, Any]:
        job = job_store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] not in TERMINAL_STATUSES:
            raise HTTPException(status_code=409, detail="Only terminal jobs can be retried")
        return submit_payload(job["parameters"])

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str, delete_files: bool = False) -> dict[str, Any]:
        job = job_store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] not in TERMINAL_STATUSES:
            raise HTTPException(status_code=409, detail="Cancel a running job before deleting it")
        deleted = job_store.delete_job(job_id)
        files_deleted = False
        if delete_files and deleted and job_store.count_jobs_for_cache_key(deleted["cache_key"]) == 0:
            shutil.rmtree(result_dir(app_config.results_dir, deleted["cache_key"]), ignore_errors=True)
            files_deleted = True
        return {"deleted": bool(deleted), "job_id": job_id, "files_deleted": files_deleted}

    @app.get("/api/storage")
    def storage() -> dict[str, Any]:
        return _storage_summary(app_config.results_dir, job_store)

    @app.post("/api/cleanup")
    def cleanup() -> dict[str, Any]:
        referenced = job_store.referenced_cache_keys()
        removed = []
        if app_config.results_dir.exists():
            for item in app_config.results_dir.iterdir():
                if item.is_dir() and KEY_RE.fullmatch(item.name) and item.name not in referenced:
                    removed.append({"cache_key": item.name, "size_bytes": _directory_size(item)})
                    shutil.rmtree(item, ignore_errors=True)
        return {"removed": removed, "storage": _storage_summary(app_config.results_dir, job_store)}

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        if job_store.get_job(job_id) is None:
            raise HTTPException(status_code=404, detail="Job not found")

        async def events():
            last_payload = ""
            while True:
                job = job_store.get_job(job_id)
                if job is None:
                    payload = {"id": job_id, "status": "missing", "terminal": True}
                else:
                    payload = job_store.public_job(job)
                data = json.dumps(payload, ensure_ascii=False)
                if data != last_payload:
                    yield f"data: {data}\n\n"
                    last_payload = data
                if payload.get("status") in TERMINAL_STATUSES or payload.get("terminal"):
                    break
                await asyncio.sleep(1.0)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/results")
    def completed_results_compat() -> dict[str, Any]:
        results = []
        for job in job_store.list_jobs():
            public = job_store.public_job(job)
            if public["status"] == "completed" and public.get("result"):
                results.append(public["result"])
        return {"results": results}

    @app.get("/results/{cache_key}/{relative:path}")
    def result_file(cache_key: str, relative: str) -> FileResponse:
        path = _resolve_result_path(app_config.results_dir, cache_key, relative)
        return FileResponse(path)

    @app.get("/viewer/{cache_key}/")
    def viewer_index(cache_key: str) -> FileResponse:
        path = _resolve_result_path(app_config.results_dir, cache_key, "webgl/index.html")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.get("/viewer/{cache_key}/{relative:path}")
    def viewer_file(cache_key: str, relative: str) -> FileResponse:
        path = _resolve_result_path(app_config.results_dir, cache_key, f"webgl/{relative}")
        return FileResponse(path)

    app.state.config = app_config
    app.state.job_store = job_store
    return app


app = build_app()


def build_parser() -> argparse.ArgumentParser:
    """Create the FastAPI app CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=load_app_config().host)
    parser.add_argument("--port", default=load_app_config().port, type=int)
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the FastAPI application with uvicorn."""

    args = build_parser().parse_args(argv)
    import uvicorn

    logging.basicConfig(level=args.log_level.upper())
    uvicorn.run("ftv_smoothing.app_server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

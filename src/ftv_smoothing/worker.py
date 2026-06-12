"""Celery worker for queued FTV smoothing calculations."""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any

try:
    from celery import Celery
except ModuleNotFoundError:  # pragma: no cover - dependency is installed in Docker
    Celery = None  # type: ignore[assignment]

from .app_core import (
    CACHE_SCHEMA_VERSION,
    ftv_config_from_params,
    has_completed_files,
    load_app_config,
    now_iso,
    parse_crop_text,
    read_result_metadata,
    result_dir,
    write_result_metadata,
)
from .job_store import JobStore
from .pipeline import orchestrate_file
from .pixel_export import export_pixel_comparison_pngs
from .roi_export import export_changed_roi_samples
from .webgl_export import WebGLTerrainConfig, export_dataset_webgl_model

LOGGER = logging.getLogger(__name__)

APP_CONFIG = load_app_config()

if Celery is not None:
    celery_app = Celery(
        "ftv_smoothing",
        broker=APP_CONFIG.celery_broker_url,
        backend=APP_CONFIG.celery_result_backend,
    )
    celery_app.conf.update(
        task_track_started=True,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
    )
else:
    class _MissingCeleryApp:
        def task(self, *args: Any, **kwargs: Any):
            def decorate(function):
                function.delay = self._missing_delay
                return function

            return decorate

        @staticmethod
        def _missing_delay(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Celery is not installed")

    celery_app = _MissingCeleryApp()


class JobCancelled(RuntimeError):
    """Raised when a queued calculation is cancelled by the user."""


def get_job_store() -> JobStore:
    """Create a job store using current environment settings."""

    config = load_app_config()
    return JobStore(config.resolved_jobs_db, config.results_dir)


def _metric_line(metrics: dict[str, Any], key: str, label: str, unit: str = "") -> str:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        suffix = f" {unit}" if unit else ""
        return f"- {label}: {value:.6g}{suffix}"
    return f"- {label}: n/a"


def write_validation_report(
    output: Path,
    *,
    params: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    """Write a concise scientific validation report for one web run."""

    lines = [
        "# FTV validation report",
        "",
        "## Parameters",
        "",
    ]
    for key in (
        "alpha",
        "k_size",
        "lambda_base",
        "max_iter",
        "tol",
        "msa_threshold",
        "backend",
        "convolution_method",
        "postprocess",
    ):
        lines.append(f"- `{key}`: {params.get(key)}")
    lines.extend(
        [
            "",
            "## Elevation correction",
            "",
            _metric_line(metrics, "residual_mean_m", "Mean residual", "m"),
            _metric_line(metrics, "residual_rmse_m", "Residual RMSE", "m"),
            _metric_line(metrics, "residual_p99_abs_m", "P99 absolute residual", "m"),
            _metric_line(metrics, "artifact_fraction_of_valid", "Artifact fraction"),
            "",
            "## Slope preservation",
            "",
            _metric_line(metrics, "slope_mean_before_deg", "Mean slope before", "deg"),
            _metric_line(metrics, "slope_mean_after_deg", "Mean slope after", "deg"),
            _metric_line(metrics, "slope_rmse_before_after_deg", "Slope RMSE before/after", "deg"),
            _metric_line(metrics, "slope_p95_abs_delta_deg", "P95 absolute slope delta", "deg"),
            _metric_line(metrics, "slope_correlation_before_after", "Slope correlation"),
            "",
            "## Curvature",
            "",
            _metric_line(metrics, "curvature_variance_before", "Curvature variance before"),
            _metric_line(metrics, "curvature_variance_after", "Curvature variance after"),
            _metric_line(metrics, "curvature_variance_ratio", "Curvature variance ratio"),
            "",
            "## Interpretation",
            "",
            "The slope comparison checks whether FTV reduces local artifacts without "
            "destroying terrain gradients. A lower curvature-variance ratio indicates "
            "smoothing, while high slope correlation indicates preservation of the "
            "large-scale relief structure.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def execute_job(job_id: str, *, store: JobStore | None = None) -> dict[str, Any]:
    """Run one FTV job synchronously; Celery calls this function."""

    config = load_app_config()
    if store is None:
        store = JobStore(config.resolved_jobs_db, config.results_dir)
    results_dir = store.results_dir

    job = store.get_job(job_id)
    if job is None:
        raise ValueError(f"Unknown job id: {job_id}")
    if job["status"] == "cancelled":
        return job.get("result_metadata") or {
            "cache_key": job["cache_key"],
            "status": "cancelled",
            "error": job.get("error") or "Cancelled by user",
        }

    params = job["parameters"]
    cache_key = job["cache_key"]

    def ensure_not_cancelled() -> None:
        if store.is_cancelled(job_id):
            raise JobCancelled("Cancelled by user")

    cached = read_result_metadata(results_dir, cache_key)
    if cached and has_completed_files(results_dir, cached):
        metadata = {**cached, "cache_hit": True}
        store.complete_job(job_id, result_metadata=metadata, stage="Loaded from saved result")
        return metadata

    directory = result_dir(results_dir, cache_key)
    output_nc = directory / "result.nc"
    comparison_png = directory / "comparison.png"
    diagnostics_png = directory / "diagnostics.png"
    slope_comparison_png = directory / "slope_comparison.png"
    pixel_dir = directory / "pixel_1to1"
    webgl_dir = directory / "webgl"
    roi_dir = directory / "roi"

    directory.mkdir(parents=True, exist_ok=True)
    ensure_not_cancelled()
    store.update_progress(
        job_id,
        status="running",
        progress_percent=1,
        stage="Preparing calculation",
    )

    def report(stage: str, progress_percent: float, details: dict[str, Any] | None = None) -> None:
        ensure_not_cancelled()
        label = stage
        if details and "iteration" in details and "max_iter" in details:
            label = f"{stage}: {details['iteration']}/{details['max_iter']}"
        store.update_progress(
            job_id,
            status="running",
            progress_percent=progress_percent,
            stage=label,
        )

    running_metadata = {
        "cache_key": cache_key,
        "created_at": job["created_at"],
        "input_fingerprint": job["input_fingerprint"],
        "parameters": params,
        "status": "running",
        "schema_version": CACHE_SCHEMA_VERSION,
    }
    write_result_metadata(results_dir, cache_key, running_metadata)

    try:
        ensure_not_cancelled()
        orchestrate_file(
            params["input_nc"],
            output_nc,
            comparison_png,
            diagnostics_png=diagnostics_png,
            slope_comparison_png=slope_comparison_png,
            config=ftv_config_from_params(params),
            crop=parse_crop_text(params["crop"]),
            progress_callback=report,
        )
        ensure_not_cancelled()
        report("Exporting 1:1 PNG artifacts", 95.5)
        pixel_exports = export_pixel_comparison_pngs(output_nc, pixel_dir)
        ensure_not_cancelled()
        report("Exporting WebGL model", 96)
        webgl = export_dataset_webgl_model(
            output_nc,
            webgl_dir,
            config=WebGLTerrainConfig(
                max_side=params["webgl_max_side"],
                vertical_exaggeration=params["vertical_exaggeration"],
            ),
        )
        ensure_not_cancelled()
        report("Exporting changed-area samples", 98)
        roi_samples = export_changed_roi_samples(
            output_nc,
            roi_dir,
            count=params.get("roi_sample_count", 4),
            config=WebGLTerrainConfig(
                max_side=params["webgl_max_side"],
                vertical_exaggeration=params["vertical_exaggeration"],
            ),
        )
        metrics_path = output_nc.with_suffix(".metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        report_path = write_validation_report(
            directory / "validation_report.md",
            params=params,
            metrics=metrics,
        )
        ensure_not_cancelled()
        metadata = {
            **running_metadata,
            "completed_at": now_iso(),
            "status": "completed",
            "cache_hit": False,
            "files": {
                "output_nc": output_nc.name,
                "comparison_png": comparison_png.name,
                "diagnostics_png": diagnostics_png.name,
                "slope_comparison_png": slope_comparison_png.name,
                "metrics_json": metrics_path.name,
                "validation_report_md": report_path.name,
                "pixel_1to1_index": "pixel_1to1/index.json",
                "webgl_index": "webgl/index.html",
                "roi_index": "roi/index.json",
            },
            "metrics": metrics,
            "webgl": webgl.metadata,
            "roi_samples": roi_samples,
            "pixel_1to1": pixel_exports,
        }
        write_result_metadata(results_dir, cache_key, metadata)
        store.complete_job(job_id, result_metadata=metadata)
        return metadata
    except JobCancelled as exc:
        cancelled = {
            **running_metadata,
            "completed_at": now_iso(),
            "status": "cancelled",
            "error": str(exc),
        }
        write_result_metadata(results_dir, cache_key, cancelled)
        store.cancel_job(job_id, reason=str(exc), result_metadata=cancelled)
        return cancelled
    except Exception as exc:
        failed = {
            **running_metadata,
            "completed_at": now_iso(),
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_result_metadata(results_dir, cache_key, failed)
        store.fail_job(job_id, str(exc), result_metadata=failed)
        raise


@celery_app.task(name="ftv_smoothing.run_ftv_job")
def run_ftv_job(job_id: str) -> dict[str, Any]:
    """Celery task entrypoint for an FTV calculation."""

    LOGGER.info("Running FTV job %s", job_id)
    return execute_job(job_id)

"""Shared settings, request parsing, and result metadata helpers for the web app."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import FTVConfig
from .webgl_export import WebGLTerrainConfig

CACHE_SCHEMA_VERSION = 3
KEY_RE = re.compile(r"^[0-9a-f]{16}$")
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
MAX_ROI_SAMPLE_COUNT = 10


@dataclass(frozen=True, slots=True)
class FTVAppConfig:
    """Runtime settings shared by the FastAPI app and Celery worker."""

    host: str = "127.0.0.1"
    port: int = 8765
    default_input_nc: Path = Path("relief_masked.nc")
    results_dir: Path = Path("artifacts/app-results")
    jobs_db: Path | None = None
    celery_broker_url: str = "redis://127.0.0.1:6379/0"
    celery_result_backend: str = "redis://127.0.0.1:6379/1"

    @property
    def resolved_jobs_db(self) -> Path:
        """Return the SQLite job database path."""

        return (self.jobs_db or self.results_dir / "jobs.sqlite3").resolve()


@dataclass(frozen=True, slots=True)
class NumericFieldSpec:
    """Server-side numeric field constraints shared with the WebUI."""

    value_type: str
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    default: float | int | None = None


_CONFIG_DEFAULTS = FTVConfig()
NUMERIC_FIELD_SPECS: dict[str, NumericFieldSpec] = {
    "alpha": NumericFieldSpec("float", 1.0, 2.0, 0.05, _CONFIG_DEFAULTS.alpha),
    "k_size": NumericFieldSpec("int", 2, 128, 1, _CONFIG_DEFAULTS.k_size),
    "lambda_base": NumericFieldSpec("float", 0.01, None, 0.01, _CONFIG_DEFAULTS.lambda_base),
    "max_iter": NumericFieldSpec("int", 1, None, 1, _CONFIG_DEFAULTS.max_iter),
    "tol": NumericFieldSpec("float", 1e-7, None, 1e-7, _CONFIG_DEFAULTS.tol),
    "msa_threshold": NumericFieldSpec("float", 0.0, None, 0.1, _CONFIG_DEFAULTS.msa_threshold),
    "interpolation_iterations": NumericFieldSpec(
        "int",
        1,
        None,
        1,
        _CONFIG_DEFAULTS.interpolation_iterations,
    ),
    "visualization_dpi": NumericFieldSpec("int", 72, 600, 12, 180),
    "webgl_max_side": NumericFieldSpec("int", 64, 2048, 64, 512),
    "vertical_exaggeration": NumericFieldSpec("float", 0.1, 8.0, 0.1, 2.4),
    "roi_sample_count": NumericFieldSpec("int", 0, MAX_ROI_SAMPLE_COUNT, 1, 10),
}


def _decimal(value: float | int | str | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def snap_numeric_value(value: float | int, spec: NumericFieldSpec) -> float | int:
    """Clamp and snap a numeric value to the nearest field step."""

    current = _decimal(value)
    assert current is not None
    minimum = _decimal(spec.min_value)
    maximum = _decimal(spec.max_value)
    step = _decimal(spec.step)

    if minimum is not None and current < minimum:
        current = minimum
    if maximum is not None and current > maximum:
        current = maximum
    if step is not None and step > 0:
        base = minimum or Decimal("0")
        units = ((current - base) / step).to_integral_value(rounding=ROUND_HALF_UP)
        current = base + units * step
        if minimum is not None and current < minimum:
            current = minimum
        if maximum is not None and current > maximum:
            current = maximum

    if spec.value_type == "int":
        return int(current.to_integral_value(rounding=ROUND_HALF_UP))
    return float(current)


def public_field_specs() -> dict[str, dict[str, float | int | str | None]]:
    """Return numeric field constraints for browser-side validation."""

    return {
        name: {
            "type": spec.value_type,
            "min": spec.min_value,
            "max": spec.max_value,
            "step": spec.step,
            "default": spec.default,
        }
        for name, spec in NUMERIC_FIELD_SPECS.items()
    }


def load_app_config() -> FTVAppConfig:
    """Read app settings from environment variables."""

    return FTVAppConfig(
        host=os.environ.get("FTV_APP_HOST", "127.0.0.1"),
        port=int(os.environ.get("FTV_APP_PORT", "8765")),
        default_input_nc=Path(os.environ.get("FTV_INPUT_NC", "relief_masked.nc")),
        results_dir=Path(os.environ.get("FTV_RESULTS_DIR", "artifacts/app-results")),
        jobs_db=Path(os.environ["FTV_JOBS_DB"]) if os.environ.get("FTV_JOBS_DB") else None,
        celery_broker_url=os.environ.get("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0"),
        celery_result_backend=os.environ.get(
            "CELERY_RESULT_BACKEND",
            "redis://127.0.0.1:6379/1",
        ),
    )


def now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def as_bool(value: Any, *, default: bool = False) -> bool:
    """Parse web-form boolean values."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def as_float(value: Any, name: str) -> float:
    """Parse a floating-point request field."""

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def as_int(value: Any, name: str) -> int:
    """Parse an integer request field."""

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def snapped_number(payload: dict[str, Any], name: str) -> float | int:
    """Parse, clamp, and snap one configured numeric request field."""

    spec = NUMERIC_FIELD_SPECS[name]
    parsed = as_float(payload.get(name, spec.default), name)
    return snap_numeric_value(parsed, spec)


def parse_crop_text(value: str | None) -> tuple[slice, slice] | None:
    """Parse a NumPy-order crop string from the web form."""

    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        row_text, col_text = text.split(",")
        row_start, row_stop = (int(part) for part in row_text.split(":"))
        col_start, col_stop = (int(part) for part in col_text.split(":"))
    except ValueError as exc:
        raise ValueError("crop must use row_start:row_stop,col_start:col_stop") from exc
    if row_start < 0 or col_start < 0 or row_stop <= row_start or col_stop <= col_start:
        raise ValueError("crop bounds must be positive and increasing")
    return slice(row_start, row_stop), slice(col_start, col_stop)


def normalize_request(payload: dict[str, Any], default_input_nc: Path) -> dict[str, Any]:
    """Validate request fields and return normalized calculation parameters."""

    input_nc = Path(str(payload.get("input_nc") or default_input_nc)).expanduser()
    crop = str(payload.get("crop") or "").strip()
    params = {
        "input_nc": str(input_nc),
        "crop": crop,
        "alpha": snapped_number(payload, "alpha"),
        "k_size": snapped_number(payload, "k_size"),
        "lambda_base": snapped_number(payload, "lambda_base"),
        "max_iter": snapped_number(payload, "max_iter"),
        "tol": snapped_number(payload, "tol"),
        "backend": str(payload.get("backend", _CONFIG_DEFAULTS.backend)),
        "convolution_method": str(
            payload.get("convolution_method", _CONFIG_DEFAULTS.convolution_method)
        ),
        "msa_threshold": snapped_number(payload, "msa_threshold"),
        "interpolation_iterations": snapped_number(payload, "interpolation_iterations"),
        "visualization_dpi": snapped_number(payload, "visualization_dpi"),
        "save_sarp": as_bool(payload.get("save_sarp"), default=False),
        "postprocess": as_bool(payload.get("postprocess"), default=True),
        "webgl_max_side": snapped_number(payload, "webgl_max_side"),
        "vertical_exaggeration": snapped_number(payload, "vertical_exaggeration"),
        "roi_sample_count": snapped_number(payload, "roi_sample_count"),
    }
    if not 0 <= params["roi_sample_count"] <= MAX_ROI_SAMPLE_COUNT:
        raise ValueError(f"roi_sample_count must be in [0, {MAX_ROI_SAMPLE_COUNT}]")
    parse_crop_text(crop)
    ftv_config_from_params(params).validate()
    WebGLTerrainConfig(
        max_side=params["webgl_max_side"],
        vertical_exaggeration=params["vertical_exaggeration"],
    ).validate()
    return params


def ftv_config_from_params(params: dict[str, Any]) -> FTVConfig:
    """Build an FTVConfig from normalized web parameters."""

    return FTVConfig(
        alpha=params["alpha"],
        k_size=params["k_size"],
        lambda_base=params["lambda_base"],
        max_iter=params["max_iter"],
        tol=params["tol"],
        backend=params["backend"],
        convolution_method=params["convolution_method"],
        msa_threshold=params["msa_threshold"],
        interpolation_iterations=params["interpolation_iterations"],
        visualization_dpi=params["visualization_dpi"],
        save_sarp=params["save_sarp"],
        postprocess=params["postprocess"],
        snapshot_every=0,
    )


def input_fingerprint(input_nc: Path) -> dict[str, Any]:
    """Fingerprint an input NetCDF file for cache invalidation."""

    resolved = input_nc.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input NetCDF not found: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_key(params: dict[str, Any], fingerprint: dict[str, Any]) -> str:
    """Return a stable cache key for input identity and all calculation parameters."""

    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "params": params,
        "input": fingerprint,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def prepare_job_request(
    payload: dict[str, Any],
    default_input_nc: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Normalize a request and return params, input fingerprint, and cache key."""

    params = normalize_request(payload, default_input_nc)
    fingerprint = input_fingerprint(Path(params["input_nc"]).expanduser())
    params = {**params, "input_nc": fingerprint["path"]}
    return params, fingerprint, cache_key(params, fingerprint)


def result_dir(results_dir: Path, key: str) -> Path:
    """Return the on-disk directory for a cache key."""

    if not KEY_RE.fullmatch(key):
        raise ValueError("Invalid result key")
    return results_dir.resolve() / key


def read_result_metadata(results_dir: Path, key: str) -> dict[str, Any] | None:
    """Read completed result metadata from disk."""

    path = result_dir(results_dir, key) / "result.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_result_metadata(results_dir: Path, key: str, metadata: dict[str, Any]) -> None:
    """Write result metadata beside generated artifacts."""

    directory = result_dir(results_dir, key)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _result_file_exists(directory: Path, relative: Any) -> bool:
    if not isinstance(relative, str) or not relative:
        return False
    target = (directory / relative).resolve()
    try:
        target.relative_to(directory)
    except ValueError:
        return False
    return target.is_file()


def _webgl_files_exist(directory: Path, relative_index: Any) -> bool:
    if not _result_file_exists(directory, relative_index):
        return False
    index_path = (directory / str(relative_index)).resolve()
    model_path = index_path.parent / "terrain-model.json"
    if not model_path.is_file():
        return False
    try:
        model = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    files = model.get("files", {})
    if not isinstance(files, dict):
        return False
    for relative in (files.get("before"), files.get("after"), files.get("mask")):
        if not _result_file_exists(index_path.parent, relative):
            return False
    if files.get("segments") and not _result_file_exists(index_path.parent, files.get("segments")):
        return False
    return True


def _roi_files_exist(directory: Path, relative_index: Any) -> bool:
    if not _result_file_exists(directory, relative_index):
        return False
    index_path = (directory / str(relative_index)).resolve()
    roi_base = index_path.parent
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        return False
    for sample in samples:
        if not isinstance(sample, dict):
            return False
        files = sample.get("files", {})
        if not isinstance(files, dict):
            return False
        for name in ("before_png", "after_png", "webgl_model"):
            if not _result_file_exists(roi_base, files.get(name)):
                return False
    return True


def has_completed_files(results_dir: Path, metadata: dict[str, Any]) -> bool:
    """Return True when required cached result files still exist."""

    key = metadata.get("cache_key")
    if not isinstance(key, str) or not KEY_RE.fullmatch(key):
        return False
    directory = result_dir(results_dir, key)
    files = metadata.get("files", {})
    required = ["output_nc", "comparison_png", "diagnostics_png", "metrics_json"]
    if int(metadata.get("schema_version", 0) or 0) >= 3:
        required.append("slope_comparison_png")
        required.append("validation_report_md")
    for name in required:
        if not _result_file_exists(directory, files.get(name)):
            return False
    if files.get("webgl_index") and not _webgl_files_exist(directory, files.get("webgl_index")):
        return False
    if files.get("roi_index") and not _roi_files_exist(directory, files.get("roi_index")):
        return False
    return True


def public_result_metadata(
    metadata: dict[str, Any],
    *,
    cache_hit: bool,
) -> dict[str, Any]:
    """Add browser URLs to result metadata."""

    key = metadata["cache_key"]
    files = metadata.get("files", {})
    urls = {
        name: f"/results/{key}/{path}"
        for name, path in files.items()
        if path and name != "webgl_index"
    }
    if files.get("webgl_index"):
        urls["webgl"] = f"/viewer/{key}/"
        urls["webgl_model"] = f"/results/{key}/webgl/terrain-model.json"
    public = dict(metadata)
    public["cache_hit"] = cache_hit
    public["urls"] = urls
    roi_samples = metadata.get("roi_samples")
    if isinstance(roi_samples, dict):
        public["roi_samples"] = _public_roi_samples(key, roi_samples)
    return public


def _public_roi_samples(key: str, roi_samples: dict[str, Any]) -> dict[str, Any]:
    public = dict(roi_samples)
    samples = []
    for sample in roi_samples.get("samples", []):
        if not isinstance(sample, dict):
            continue
        item = dict(sample)
        files = item.get("files", {})
        if isinstance(files, dict):
            item["urls"] = {
                name: f"/results/{key}/roi/{path}"
                for name, path in files.items()
                if isinstance(path, str) and path
            }
        samples.append(item)
    public["samples"] = samples
    return public

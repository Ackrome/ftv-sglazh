"""End-to-end FTV orchestration."""

from __future__ import annotations

import gc
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
import xarray as xr

from .artifacts import detect_artifacts_msa
from .config import FTVConfig
from .io import load_and_prepare_ds, open_dataset, restore_nan_topology
from .metrics import evaluate_no_reference_metrics
from .postprocess import morphological_postprocess
from .solver import FTVResult, fractional_total_variation_denoise
from .visualize import visualize_before_after, visualize_diagnostics, visualize_slope_comparison

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[str, float, dict[str, Any] | None], None]


def _coordinates(ds: xr.Dataset) -> tuple[np.ndarray | None, np.ndarray | None]:
    longitude = np.asarray(ds["longitude"].values) if "longitude" in ds.coords else None
    latitude = np.asarray(ds["latitude"].values) if "latitude" in ds.coords else None
    return longitude, latitude


def _report_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    progress_percent: float,
    **details: Any,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, progress_percent, details or None)


def _snapshot_callback(
    original: np.ndarray,
    invalid_mask: np.ndarray,
    snapshot_dir: str | Path | None,
    longitude: np.ndarray | None,
    latitude: np.ndarray | None,
    dpi: int,
):
    if snapshot_dir is None:
        return None
    directory = Path(snapshot_dir)
    directory.mkdir(parents=True, exist_ok=True)

    def save(iteration: int, current: np.ndarray, error: float) -> None:
        current = restore_nan_topology(current, invalid_mask)
        visualize_before_after(
            restore_nan_topology(original, invalid_mask),
            current,
            directory / f"iteration_{iteration:04d}.png",
            longitude=longitude,
            latitude=latitude,
            dpi=dpi,
            title=f"FTV intermediate comparison: iteration {iteration}, error {error:.3e}",
        )

    return save


def _write_dataset(
    ds: xr.Dataset,
    output_nc: str | Path,
    corrected: np.ndarray,
    artifact_mask: np.ndarray,
    fidelity_weight: np.ndarray,
    config: FTVConfig,
    result: FTVResult,
    metrics: dict[str, float],
) -> xr.Dataset:
    output = Path(output_nc)
    output.parent.mkdir(parents=True, exist_ok=True)
    dims = ds["reprojected_dem"].dims
    out = ds.copy(deep=False)
    out["reprojected_dem_ftv"] = xr.DataArray(
        corrected.astype(np.float32, copy=False),
        dims=dims,
        attrs={
            "long_name": "FTV-corrected digital elevation model",
            "units": ds["reprojected_dem"].attrs.get("units", "m"),
            "ftv_alpha": config.alpha,
            "ftv_k_size": config.k_size,
            "ftv_backend": result.backend,
        },
    )
    out["ftv_artifact_mask"] = xr.DataArray(
        artifact_mask.astype(np.int8),
        dims=dims,
        attrs={"long_name": "MSA artifact mask", "flag_values": "0, 1"},
    )
    if config.save_sarp:
        out["ftv_lambda_sarp"] = xr.DataArray(
            fidelity_weight.astype(np.float32, copy=False),
            dims=dims,
            attrs={"long_name": "FTV spatially adaptive data fidelity weight"},
        )
    out.attrs.update(
        {
            "ftv_smoothing": "fractional-order total variation with Chambolle-Pock PDHG",
            "ftv_config_json": json.dumps(config.to_dict(), sort_keys=True),
            "ftv_backend": result.backend,
            "ftv_iterations": result.iterations,
            "ftv_converged": int(result.converged),
            "ftv_elapsed_seconds": result.elapsed_seconds,
            "ftv_metrics_json": json.dumps(metrics, sort_keys=True),
        }
    )
    encoding: dict[str, dict[str, Any]] = {
        "reprojected_dem_ftv": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "ftv_artifact_mask": {"zlib": True, "complevel": 4, "dtype": "int8"},
    }
    if config.save_sarp:
        encoding["ftv_lambda_sarp"] = {"zlib": True, "complevel": 4, "dtype": "float32"}
    LOGGER.info("Writing output NetCDF to %s", output)
    out.to_netcdf(output, engine="h5netcdf", encoding=encoding)
    return out


def orchestrate_denoising(
    ds: xr.Dataset,
    output_nc: str | Path,
    output_png: str | Path | None = None,
    *,
    diagnostics_png: str | Path | None = None,
    slope_comparison_png: str | Path | None = None,
    forge3d_png: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
    config: FTVConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> xr.Dataset:
    """Execute the complete FTV processing graph and serialize its products."""

    if config is None:
        config = FTVConfig()
    config.validate()
    LOGGER.info("FTV orchestration started")
    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = time.perf_counter()

    _report_progress(progress_callback, "Loading and preprocessing DEM", 3)
    dem, invalid_mask, derivatives = load_and_prepare_ds(
        ds,
        interpolation_iterations=config.interpolation_iterations,
    )
    _report_progress(progress_callback, "Preprocessing complete", 10)
    valid_mask = ~invalid_mask
    longitude, latitude = _coordinates(ds)
    _report_progress(progress_callback, "Detecting MSA artifacts", 12)
    artifact_mask = detect_artifacts_msa(
        derivatives["slope"],
        threshold=config.msa_threshold,
        window=config.msa_window,
        robust_sigma=config.msa_robust_sigma,
        dilation_iterations=config.msa_dilation_iterations,
        valid_mask=valid_mask,
    )
    _report_progress(progress_callback, "Artifact detection complete", 15)
    callback = _snapshot_callback(
        dem,
        invalid_mask,
        snapshot_dir,
        longitude,
        latitude,
        config.snapshot_dpi,
    )

    def report_optimizer(iteration: int, max_iter: int, error: float | None) -> None:
        progress = 15.0 + 60.0 * min(1.0, iteration / max(1, max_iter))
        _report_progress(
            progress_callback,
            "Optimizing FTV",
            progress,
            iteration=iteration,
            max_iter=max_iter,
            relative_error=error,
        )

    result = fractional_total_variation_denoise(
        dem,
        artifact_mask,
        derivatives["slope"],
        max_curvature=derivatives["max_curvature"],
        config=config,
        valid_mask=valid_mask,
        snapshot_callback=callback,
        progress_callback=report_optimizer,
        return_result=True,
    )
    assert isinstance(result, FTVResult)
    _report_progress(progress_callback, "Optimization complete", 75)
    corrected = result.corrected_dem
    if config.postprocess:
        _report_progress(progress_callback, "Postprocessing terrain", 77)
        corrected = morphological_postprocess(
            corrected,
            dem,
            valid_mask=valid_mask,
            max_component_size=config.postprocess_component_size,
            diffusion_iterations=config.postprocess_diffusion_iterations,
            kappa=config.postprocess_kappa,
            gamma=config.postprocess_gamma,
        )
    _report_progress(progress_callback, "Evaluating metrics", 82)
    corrected_with_nan = restore_nan_topology(corrected, invalid_mask)
    original_with_nan = restore_nan_topology(dem, invalid_mask)
    resolution_m = float(ds.attrs.get("resolution_m", 30.0))
    metrics = evaluate_no_reference_metrics(
        dem,
        corrected,
        valid_mask,
        resolution_m=resolution_m,
    )
    metrics.update(
        {
            "optimizer_iterations": float(result.iterations),
            "optimizer_elapsed_seconds": result.elapsed_seconds,
            "artifact_cells": float(artifact_mask.sum()),
            "artifact_fraction_of_valid": float(artifact_mask.sum() / max(1, valid_mask.sum())),
            "pipeline_elapsed_seconds_before_serialization": time.perf_counter() - started,
            "rss_before_bytes": float(rss_before),
            "rss_peak_observed_bytes": float(process.memory_info().rss),
        }
    )
    metrics_path = Path(output_nc).with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote metrics to %s", metrics_path)

    if output_png is not None:
        _report_progress(progress_callback, "Rendering comparison PNG", 86)
        LOGGER.info("Writing final before/after PNG to %s", output_png)
        visualize_before_after(
            original_with_nan,
            corrected_with_nan,
            output_png,
            longitude=longitude,
            latitude=latitude,
            dpi=config.visualization_dpi,
        )
    if diagnostics_png is not None:
        _report_progress(progress_callback, "Rendering diagnostics PNG", 88)
        LOGGER.info("Writing diagnostics PNG to %s", diagnostics_png)
        visualize_diagnostics(
            dem,
            corrected,
            diagnostics_png,
            valid_mask=valid_mask,
            longitude=longitude,
            latitude=latitude,
            resolution_m=resolution_m,
        )
    if slope_comparison_png is not None:
        _report_progress(progress_callback, "Rendering slope comparison PNG", 90)
        LOGGER.info("Writing slope comparison PNG to %s", slope_comparison_png)
        visualize_slope_comparison(
            dem,
            corrected,
            slope_comparison_png,
            valid_mask=valid_mask,
            longitude=longitude,
            latitude=latitude,
            resolution_m=resolution_m,
            dpi=config.visualization_dpi,
        )
    if forge3d_png is not None:
        _report_progress(progress_callback, "Rendering Forge3D comparison", 92)
        from .forge3d_visualize import Forge3DConfig, render_forge3d_comparison

        LOGGER.info("Writing Forge3D terrain comparison to %s", forge3d_png)
        render_forge3d_comparison(
            original_with_nan,
            corrected_with_nan,
            forge3d_png,
            config=Forge3DConfig(resolution_m=resolution_m),
            source_label=Path(output_nc).name,
        )
    _report_progress(progress_callback, "Writing NetCDF result", 94)
    out = _write_dataset(
        ds,
        output_nc,
        corrected_with_nan,
        artifact_mask,
        result.fidelity_weight,
        config,
        result,
        metrics,
    )
    derivatives.clear()
    gc.collect()
    _report_progress(progress_callback, "Pipeline complete", 95)
    LOGGER.info("FTV orchestration completed")
    return out


def orchestrate_file(
    input_nc: str | Path,
    output_nc: str | Path,
    output_png: str | Path | None = None,
    *,
    diagnostics_png: str | Path | None = None,
    slope_comparison_png: str | Path | None = None,
    forge3d_png: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
    config: FTVConfig | None = None,
    crop: tuple[slice, slice] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> xr.Dataset:
    """Open a NetCDF file, optionally crop it, and run orchestration."""

    LOGGER.info(
        "Opening %s on %s / Python %s",
        input_nc,
        platform.platform(),
        platform.python_version(),
    )
    with open_dataset(input_nc) as ds:
        working = ds
        if crop is not None:
            y_slice, x_slice = crop
            working = ds.isel(latitude=y_slice, longitude=x_slice)
            LOGGER.info("Using ROI crop latitude=%s longitude=%s", y_slice, x_slice)
        return orchestrate_denoising(
            working,
            output_nc,
            output_png,
            diagnostics_png=diagnostics_png,
            slope_comparison_png=slope_comparison_png,
            forge3d_png=forge3d_png,
            snapshot_dir=snapshot_dir,
            config=config,
            progress_callback=progress_callback,
        )

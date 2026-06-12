from pathlib import Path

import numpy as np
import xarray as xr

from ftv_smoothing.artifacts import detect_artifacts_msa
from ftv_smoothing.config import FTVConfig
from ftv_smoothing.io import load_and_prepare_ds
from ftv_smoothing.metrics import evaluate_no_reference_metrics
from ftv_smoothing.pipeline import orchestrate_denoising
from ftv_smoothing.roi_export import select_changed_rois
from ftv_smoothing.solver import fractional_total_variation_denoise


def _dataset(size: int = 40) -> xr.Dataset:
    y = np.linspace(43.0, 42.9, size)
    x = np.linspace(130.0, 130.1, size)
    yy, xx = np.mgrid[:size, :size]
    dem = (100 + 0.5 * xx + 0.2 * yy).astype(np.float32)
    slope = np.full_like(dem, 2.0)
    curvature = np.zeros_like(dem)
    return xr.Dataset(
        {
            "reprojected_dem": (("latitude", "longitude"), dem),
            "slope": (("latitude", "longitude"), slope),
            "max_curvature": (("latitude", "longitude"), curvature),
        },
        coords={"latitude": y, "longitude": x},
        attrs={"resolution_m": 30},
    )


def test_load_prepare_fills_working_dem_and_preserves_mask() -> None:
    ds = _dataset()
    ds["reprojected_dem"].values[10:13, 15:18] = np.nan
    dem, invalid, derivatives = load_and_prepare_ds(ds, interpolation_iterations=3)
    assert invalid.sum() == 9
    assert np.isfinite(dem).all()
    assert dem.dtype == np.float32
    assert derivatives["slope"].dtype == np.float32


def test_msa_detects_local_outlier() -> None:
    slope = np.zeros((31, 31), dtype=np.float32)
    slope[15, 15] = 100.0
    mask = detect_artifacts_msa(slope, threshold=5.0, dilation_iterations=1)
    assert mask[15, 15]
    assert mask.sum() > 1


def test_ftv_reduces_plane_noise() -> None:
    rng = np.random.default_rng(9)
    yy, xx = np.mgrid[:64, :64]
    plane = (100 + 0.25 * xx + 0.1 * yy).astype(np.float32)
    noisy = plane + rng.normal(0, 1.5, plane.shape).astype(np.float32)
    slope = np.full_like(plane, 1.0)
    corrected = fractional_total_variation_denoise(
        noisy,
        np.zeros_like(noisy, dtype=np.bool_),
        slope,
        config=FTVConfig(
            backend="cpu",
            alpha=1.5,
            k_size=10,
            lambda_base=0.35,
            max_iter=80,
            tol=1e-6,
            log_every=10,
            postprocess=False,
        ),
    )
    noisy_rmse = np.sqrt(np.mean((noisy - plane) ** 2))
    corrected_rmse = np.sqrt(np.mean((corrected - plane) ** 2))
    assert corrected_rmse < noisy_rmse


def test_slope_delta_metrics_are_reported() -> None:
    yy, xx = np.mgrid[:24, :24]
    before = (100 + 0.2 * xx + 0.1 * yy).astype(np.float32)
    after = before.copy()
    after[8:16, 8:16] += np.linspace(0, 3, 8, dtype=np.float32)[None, :]
    metrics = evaluate_no_reference_metrics(
        before,
        after,
        np.ones_like(before, dtype=np.bool_),
        resolution_m=30.0,
    )

    assert metrics["slope_rmse_before_after_deg"] > 0
    assert metrics["slope_p95_abs_delta_deg"] > 0
    assert -1 <= metrics["slope_correlation_before_after"] <= 1


def test_fractional_order_reduces_staircasing_vs_classic_tv() -> None:
    rng = np.random.default_rng(19)
    yy, xx = np.mgrid[:96, :96]
    plane = (50 + 0.12 * xx + 0.07 * yy).astype(np.float32)
    noisy = plane + rng.normal(0, 1.2, plane.shape).astype(np.float32)
    slope = np.ones_like(plane)
    artifact_mask = np.zeros_like(plane, dtype=np.bool_)

    def run(alpha: float) -> np.ndarray:
        return fractional_total_variation_denoise(
            noisy,
            artifact_mask,
            slope,
            config=FTVConfig(
                backend="cpu",
                alpha=alpha,
                k_size=12,
                lambda_base=0.3,
                max_iter=120,
                tol=1e-7,
                log_every=20,
                postprocess=False,
            ),
        )

    def second_derivative_variance(array: np.ndarray) -> float:
        d2x = np.diff(array, n=2, axis=1)
        d2y = np.diff(array, n=2, axis=0)
        return float(np.mean(d2x * d2x) + np.mean(d2y * d2y))

    classic_tv = run(1.0)
    fractional_tv = run(1.5)
    assert second_derivative_variance(fractional_tv) < second_derivative_variance(classic_tv)


def test_orchestrator_restores_nan_identity(tmp_path: Path) -> None:
    ds = _dataset(24)
    ds["reprojected_dem"].values[3:5, 7:9] = np.nan
    output = tmp_path / "result.nc"
    result = orchestrate_denoising(
        ds,
        output,
        config=FTVConfig(
            backend="cpu",
            alpha=1.4,
            k_size=10,
            lambda_base=0.5,
            max_iter=8,
            interpolation_iterations=2,
            log_every=2,
            postprocess=False,
        ),
    )
    assert output.exists()
    assert np.isnan(result["reprojected_dem_ftv"].values[3:5, 7:9]).all()
    assert np.isfinite(result["reprojected_dem_ftv"].values[0, 0])


def test_select_changed_rois_prefers_non_overlapping_large_changes() -> None:
    before = np.zeros((40, 40), dtype=np.float32)
    after = before.copy()
    after[4:12, 5:13] = 5.0
    after[24:32, 25:33] = 3.0

    samples = select_changed_rois(before, after, count=2, window_size=8)

    assert len(samples) == 2
    assert samples[0].mean_abs_change_m > samples[1].mean_abs_change_m
    assert 4 <= samples[0].row_start <= 6
    assert 5 <= samples[0].col_start <= 7
    assert 24 <= samples[1].row_start <= 26
    assert 25 <= samples[1].col_start <= 27

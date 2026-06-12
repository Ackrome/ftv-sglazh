"""No-reference terrain metrics for before/after validation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi


def derive_slope_and_curvature(
    dem: np.ndarray,
    resolution_m: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope in degrees and a Laplacian curvature proxy."""

    dem = np.asarray(dem, dtype=np.float32)
    grad_y, grad_x = np.gradient(dem, resolution_m)
    slope = np.degrees(np.arctan(np.hypot(grad_x, grad_y))).astype(np.float32)
    curvature = ndi.laplace(dem, mode="nearest") / np.float32(resolution_m**2)
    return slope.astype(np.float32), curvature.astype(np.float32)


def _float(value: Any) -> float:
    return float(np.asarray(value).item())


def evaluate_no_reference_metrics(
    original: np.ndarray,
    corrected: np.ndarray,
    valid_mask: np.ndarray,
    *,
    resolution_m: float = 30.0,
) -> dict[str, float]:
    """Calculate terrain-preservation and smoothing indicators."""

    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    slope_before, curvature_before = derive_slope_and_curvature(original, resolution_m)
    slope_after, curvature_after = derive_slope_and_curvature(corrected, resolution_m)
    residual = corrected - original
    before = slope_before[valid_mask]
    after = slope_after[valid_mask]
    slope_delta = after - before
    curv_before = curvature_before[valid_mask]
    curv_after = curvature_after[valid_mask]
    valid_residual = residual[valid_mask]
    before_std = np.std(before, dtype=np.float64)
    after_std = np.std(after, dtype=np.float64)
    if before_std > 0 and after_std > 0:
        slope_correlation = np.corrcoef(before.astype(np.float64), after.astype(np.float64))[0, 1]
    else:
        slope_correlation = 1.0 if np.allclose(before, after) else 0.0
    return {
        "valid_cells": float(valid_mask.sum()),
        "residual_mean_m": _float(np.mean(valid_residual, dtype=np.float64)),
        "residual_std_m": _float(np.std(valid_residual, dtype=np.float64)),
        "residual_rmse_m": _float(np.sqrt(np.mean(valid_residual.astype(np.float64) ** 2))),
        "residual_p99_abs_m": _float(np.percentile(np.abs(valid_residual), 99)),
        "slope_mean_before_deg": _float(np.mean(before, dtype=np.float64)),
        "slope_mean_after_deg": _float(np.mean(after, dtype=np.float64)),
        "slope_std_before_deg": _float(before_std),
        "slope_std_after_deg": _float(after_std),
        "slope_mean_abs_delta_deg": _float(np.mean(np.abs(slope_delta), dtype=np.float64)),
        "slope_rmse_before_after_deg": _float(
            np.sqrt(np.mean(slope_delta.astype(np.float64) ** 2))
        ),
        "slope_p95_abs_delta_deg": _float(np.percentile(np.abs(slope_delta), 95)),
        "slope_correlation_before_after": _float(slope_correlation),
        "slope_near_zero_before_fraction": _float(np.mean(before < 0.1)),
        "slope_near_zero_after_fraction": _float(np.mean(after < 0.1)),
        "curvature_variance_before": _float(np.var(curv_before, dtype=np.float64)),
        "curvature_variance_after": _float(np.var(curv_after, dtype=np.float64)),
        "curvature_variance_ratio": _float(
            np.var(curv_after, dtype=np.float64)
            / max(np.var(curv_before, dtype=np.float64), 1e-12)
        ),
    }

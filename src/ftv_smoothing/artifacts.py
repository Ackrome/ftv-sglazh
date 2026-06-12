"""MSA artifact detection and spatially adaptive fidelity weights."""

from __future__ import annotations

import logging

import numpy as np
from numba import njit, prange
from scipy import ndimage as ndi

LOGGER = logging.getLogger(__name__)


def detect_artifacts_msa(
    slope_array: np.ndarray,
    threshold: float = 5.0,
    window: int = 7,
    robust_sigma: float = 5.0,
    dilation_iterations: int = 2,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Detect local slope anomalies using a robust Maximum Slope Approach.

    A local mean removes broad natural terrain trends. The robust threshold is
    the larger of the user threshold and a global MAD-derived residual limit.
    The resulting mask is expanded to include anomaly boundaries.
    """

    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")
    slope = np.asarray(slope_array, dtype=np.float32)
    if slope.ndim != 2:
        raise ValueError("slope_array must be two-dimensional")
    if valid_mask is None:
        valid_mask = np.isfinite(slope)
    else:
        valid_mask = np.asarray(valid_mask, dtype=np.bool_)

    LOGGER.info("Running MSA artifact detection with threshold %.3f", threshold)
    local_mean = ndi.uniform_filter(slope, size=window, mode="nearest")
    residual = np.abs(slope - local_mean)
    values = residual[valid_mask]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_limit = median + robust_sigma * 1.4826 * max(mad, 1e-6)
    adaptive_limit = max(float(threshold), robust_limit)
    artifact_mask = (residual > adaptive_limit) & valid_mask
    if dilation_iterations:
        artifact_mask = ndi.binary_dilation(
            artifact_mask,
            iterations=dilation_iterations,
        )
        artifact_mask &= valid_mask
    LOGGER.info(
        "MSA threshold %.3f selected; %d cells marked (%.3f%% of valid cells)",
        adaptive_limit,
        int(artifact_mask.sum()),
        float(100 * artifact_mask.sum() / max(1, valid_mask.sum())),
    )
    return artifact_mask.astype(np.bool_, copy=False)


@njit(parallel=True, fastmath=True, cache=True)
def _generate_sarp_numba(
    slope: np.ndarray,
    curvature: np.ndarray,
    artifact_mask: np.ndarray,
    lambda_base: float,
    slope_scale: float,
    curvature_scale: float,
    slope_gain: float,
    curvature_gain: float,
    artifact_fidelity_scale: float,
) -> np.ndarray:
    rows, cols = slope.shape
    result = np.empty((rows, cols), dtype=np.float32)
    floor = lambda_base * 0.05
    for row in prange(rows):
        for col in range(cols):
            slope_norm = min(abs(slope[row, col]) / slope_scale, 1.0)
            curv_norm = min(abs(curvature[row, col]) / curvature_scale, 1.0)
            weight = lambda_base * (
                1.0 + slope_gain * slope_norm + curvature_gain * curv_norm
            )
            if artifact_mask[row, col]:
                weight *= artifact_fidelity_scale
            result[row, col] = max(weight, floor)
    return result


def generate_sarp_weights(
    slope: np.ndarray,
    max_curvature: np.ndarray,
    artifact_mask: np.ndarray,
    lambda_base: float = 1.0,
    slope_gain: float = 3.0,
    curvature_gain: float = 2.0,
    artifact_fidelity_scale: float = 0.25,
    clip_percentile: float = 95.0,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Generate SARP data-fidelity weights from slope and curvature.

    High terrain gradients receive higher fidelity to preserve ridges and
    scarps. MSA artifacts receive lower fidelity so FTV can correct them more
    aggressively.
    """

    slope = np.asarray(slope, dtype=np.float32)
    curvature = np.asarray(max_curvature, dtype=np.float32)
    artifact_mask = np.asarray(artifact_mask, dtype=np.bool_)
    if slope.shape != curvature.shape or slope.shape != artifact_mask.shape:
        raise ValueError("slope, max_curvature, and artifact_mask shapes must match")
    if valid_mask is None:
        valid_mask = np.ones(slope.shape, dtype=np.bool_)

    slope_scale = max(float(np.percentile(np.abs(slope[valid_mask]), clip_percentile)), 1e-6)
    curvature_scale = max(
        float(np.percentile(np.abs(curvature[valid_mask]), clip_percentile)),
        1e-6,
    )
    LOGGER.info(
        "Generating SARP weights (slope scale %.4f, curvature scale %.4f)",
        slope_scale,
        curvature_scale,
    )
    return _generate_sarp_numba(
        slope,
        curvature,
        artifact_mask,
        float(lambda_base),
        slope_scale,
        curvature_scale,
        float(slope_gain),
        float(curvature_gain),
        float(artifact_fidelity_scale),
    )


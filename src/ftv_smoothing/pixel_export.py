"""Export exact-grid PNG artifacts for downloaded web results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .io import open_dataset
from .metrics import derive_slope_and_curvature


def _limits(values: np.ndarray, low: float = 0.5, high: float = 99.5) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, low))
    vmax = float(np.percentile(finite, high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        center = float(np.mean(finite))
        return center - 0.5, center + 0.5
    return vmin, vmax


def _symmetric_limit(values: np.ndarray, percentile: float = 99.0) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    return max(float(np.percentile(finite, percentile)), 1e-6)


def _save_colormapped_png(
    data: np.ndarray,
    output: Path,
    *,
    cmap_name: str,
    vmin: float,
    vmax: float,
) -> None:
    array = np.asarray(data, dtype=np.float32)
    valid = np.isfinite(array)
    norm = np.zeros(array.shape, dtype=np.float32)
    if vmax > vmin:
        norm[valid] = np.clip((array[valid] - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = (plt.get_cmap(cmap_name)(norm) * 255).astype(np.uint8)
    rgba[~valid] = np.array([255, 255, 255, 255], dtype=np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(output)


def export_pixel_comparison_pngs(
    input_nc: str | Path,
    output_dir: str | Path,
    *,
    before_var: str = "reprojected_dem",
    after_var: str = "reprojected_dem_ftv",
) -> dict[str, Any]:
    """Export before/after/delta PNGs with one DEM cell per image pixel."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open_dataset(input_nc) as ds:
        before = np.asarray(ds[before_var].values, dtype=np.float32)
        after = np.asarray(ds[after_var].values, dtype=np.float32)
        resolution_m = float(ds.attrs.get("resolution_m", 30.0))

    if before.ndim != 2 or before.shape != after.shape:
        raise ValueError("1:1 PNG export requires paired 2D arrays with the same shape")

    valid = np.isfinite(before) & np.isfinite(after)
    difference = np.where(valid, after - before, np.nan).astype(np.float32)
    slope_before, _ = derive_slope_and_curvature(np.where(np.isfinite(before), before, 0), resolution_m)
    slope_after, _ = derive_slope_and_curvature(np.where(np.isfinite(after), after, 0), resolution_m)
    slope_before = np.where(valid, slope_before, np.nan).astype(np.float32)
    slope_after = np.where(valid, slope_after, np.nan).astype(np.float32)
    slope_delta = np.where(valid, slope_after - slope_before, np.nan).astype(np.float32)

    elevation_limits = _limits(np.concatenate([before[np.isfinite(before)], after[np.isfinite(after)]]))
    delta_limit = _symmetric_limit(difference)
    slope_limits = _limits(
        np.concatenate([slope_before[np.isfinite(slope_before)], slope_after[np.isfinite(slope_after)]])
    )
    slope_delta_limit = _symmetric_limit(slope_delta)

    specs = {
        "elevation_before_png": (before, "terrain", elevation_limits[0], elevation_limits[1]),
        "elevation_after_png": (after, "terrain", elevation_limits[0], elevation_limits[1]),
        "elevation_delta_png": (difference, "coolwarm", -delta_limit, delta_limit),
        "slope_before_png": (slope_before, "viridis", slope_limits[0], slope_limits[1]),
        "slope_after_png": (slope_after, "viridis", slope_limits[0], slope_limits[1]),
        "slope_delta_png": (slope_delta, "coolwarm", -slope_delta_limit, slope_delta_limit),
    }

    files: dict[str, str] = {}
    for key, (data, cmap_name, vmin, vmax) in specs.items():
        filename = key.replace("_png", "_1to1.png")
        _save_colormapped_png(data, output / filename, cmap_name=cmap_name, vmin=vmin, vmax=vmax)
        files[key] = filename

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "pixel_mapping": "1 DEM grid cell = 1 PNG pixel",
        "shape": [int(before.shape[0]), int(before.shape[1])],
        "resolution_m": resolution_m,
        "files": files,
        "color_limits": {
            "elevation_m": list(elevation_limits),
            "elevation_delta_m": [-delta_limit, delta_limit],
            "slope_deg": list(slope_limits),
            "slope_delta_deg": [-slope_delta_limit, slope_delta_limit],
        },
    }
    (output / "index.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata

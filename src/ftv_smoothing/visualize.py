"""Portable PNG reporting for FTV comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from .metrics import derive_slope_and_curvature


def _extent(
    longitude: np.ndarray | None,
    latitude: np.ndarray | None,
    shape: tuple[int, int],
) -> list[float]:
    if longitude is None or latitude is None:
        return [0.0, float(shape[1]), 0.0, float(shape[0])]
    return [
        float(np.min(longitude)),
        float(np.max(longitude)),
        float(np.min(latitude)),
        float(np.max(latitude)),
    ]


def _add_scale_bar(
    ax: plt.Axes,
    extent: Sequence[float],
    latitude: np.ndarray | None,
) -> None:
    if latitude is None:
        return
    mean_lat_rad = np.deg2rad(float(np.mean(latitude)))
    km_per_degree_lon = 111.32 * max(float(np.cos(mean_lat_rad)), 1e-3)
    width_km = abs(extent[1] - extent[0]) * km_per_degree_lon
    target = max(width_km / 5.0, 0.1)
    choices = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
    scale_km = float(choices[np.argmin(np.abs(choices - target))])
    bar = AnchoredSizeBar(
        ax.transData,
        scale_km / km_per_degree_lon,
        f"{scale_km:g} km",
        "lower right",
        pad=0.4,
        color="black",
        frameon=True,
        size_vertical=max((extent[3] - extent[2]) * 0.003, 1e-6),
    )
    ax.add_artist(bar)


def _topography_limits(original: np.ndarray, corrected: np.ndarray) -> tuple[float, float]:
    values = np.concatenate(
        [original[np.isfinite(original)], corrected[np.isfinite(corrected)]]
    )
    return float(np.percentile(values, 0.5)), float(np.percentile(values, 99.5))


def visualize_before_after(
    original_dem: np.ndarray,
    corrected_dem: np.ndarray,
    output_png: str | Path,
    *,
    longitude: np.ndarray | None = None,
    latitude: np.ndarray | None = None,
    dpi: int = 600,
    title: str = "Fractional-order TV DEM smoothing",
) -> Path:
    """Save a shared-scale topographic before/after PNG comparison."""

    original = np.asarray(original_dem, dtype=np.float32)
    corrected = np.asarray(corrected_dem, dtype=np.float32)
    if original.shape != corrected.shape:
        raise ValueError("original_dem and corrected_dem shapes must match")
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    extent = _extent(longitude, latitude, original.shape)
    vmin, vmax = _topography_limits(original, corrected)
    cmap = plt.get_cmap("terrain").copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    images = []
    for ax, data, panel_title in zip(
        axes,
        (original, corrected),
        ("Original reprojected_dem", "FTV reprojected_dem"),
    ):
        images.append(
            ax.imshow(
                data,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                origin="upper",
                interpolation="nearest",
            )
        )
        ax.set_title(panel_title)
        ax.set_xlabel("longitude" if longitude is not None else "column")
        ax.set_ylabel("latitude" if latitude is not None else "row")
        _add_scale_bar(ax, extent, latitude)
    fig.colorbar(images[0], ax=axes, shrink=0.88, label="Elevation, m")
    fig.suptitle(title, fontsize=14)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def visualize_diagnostics(
    original_dem: np.ndarray,
    corrected_dem: np.ndarray,
    output_png: str | Path,
    *,
    valid_mask: np.ndarray,
    longitude: np.ndarray | None = None,
    latitude: np.ndarray | None = None,
    resolution_m: float = 30.0,
    dpi: int = 300,
) -> Path:
    """Save residual and slope-distribution diagnostics."""

    original = np.asarray(original_dem, dtype=np.float32)
    corrected = np.asarray(corrected_dem, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    extent = _extent(longitude, latitude, original.shape)
    difference = np.where(valid_mask, corrected - original, np.nan)
    diff_limit = max(float(np.nanpercentile(np.abs(difference), 99)), 1e-3)
    slope_before, _ = derive_slope_and_curvature(original, resolution_m)
    slope_after, _ = derive_slope_and_curvature(corrected, resolution_m)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    diff_cmap = plt.get_cmap("coolwarm").copy()
    diff_cmap.set_bad("white")
    image = axes[0].imshow(
        difference,
        extent=extent,
        origin="upper",
        cmap=diff_cmap,
        vmin=-diff_limit,
        vmax=diff_limit,
        interpolation="nearest",
    )
    axes[0].set_title("FTV correction: after - before")
    axes[0].set_xlabel("longitude" if longitude is not None else "column")
    axes[0].set_ylabel("latitude" if latitude is not None else "row")
    _add_scale_bar(axes[0], extent, latitude)
    fig.colorbar(image, ax=axes[0], shrink=0.88, label="Elevation correction, m")

    max_slope = float(np.percentile(slope_before[valid_mask], 99.5))
    bins = np.linspace(0, max(max_slope, 1e-3), 100)
    axes[1].hist(
        slope_before[valid_mask],
        bins=bins,
        density=True,
        alpha=0.55,
        label="before",
    )
    axes[1].hist(
        slope_after[valid_mask],
        bins=bins,
        density=True,
        alpha=0.55,
        label="after",
    )
    axes[1].set_title("Slope consistency")
    axes[1].set_xlabel("Derived slope, degrees")
    axes[1].set_ylabel("Probability density")
    axes[1].legend()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def visualize_slope_comparison(
    original_dem: np.ndarray,
    corrected_dem: np.ndarray,
    output_png: str | Path,
    *,
    valid_mask: np.ndarray,
    longitude: np.ndarray | None = None,
    latitude: np.ndarray | None = None,
    resolution_m: float = 30.0,
    dpi: int = 300,
) -> Path:
    """Save slope before/after maps and a slope-delta panel."""

    original = np.asarray(original_dem, dtype=np.float32)
    corrected = np.asarray(corrected_dem, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    if original.shape != corrected.shape or original.shape != valid_mask.shape:
        raise ValueError("slope comparison arrays must have matching shapes")

    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    extent = _extent(longitude, latitude, original.shape)
    slope_before, _ = derive_slope_and_curvature(original, resolution_m)
    slope_after, _ = derive_slope_and_curvature(corrected, resolution_m)
    slope_before = np.where(valid_mask, slope_before, np.nan)
    slope_after = np.where(valid_mask, slope_after, np.nan)
    slope_delta = np.where(valid_mask, slope_after - slope_before, np.nan)

    valid_before = slope_before[np.isfinite(slope_before)]
    slope_max = float(np.percentile(valid_before, 99.5)) if valid_before.size else 1.0
    delta_values = np.abs(slope_delta[np.isfinite(slope_delta)])
    delta_limit = float(np.percentile(delta_values, 99.0)) if delta_values.size else 1.0
    slope_max = max(slope_max, 1e-3)
    delta_limit = max(delta_limit, 1e-3)

    slope_cmap = plt.get_cmap("viridis").copy()
    slope_cmap.set_bad("white")
    delta_cmap = plt.get_cmap("coolwarm").copy()
    delta_cmap.set_bad("white")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    panels = [
        (slope_before, "Slope before FTV", slope_cmap, 0.0, slope_max, "Slope, degrees"),
        (slope_after, "Slope after FTV", slope_cmap, 0.0, slope_max, "Slope, degrees"),
        (
            slope_delta,
            "Slope delta: after - before",
            delta_cmap,
            -delta_limit,
            delta_limit,
            "Slope change, degrees",
        ),
    ]
    for ax, (data, title, cmap, vmin, vmax, label) in zip(axes, panels):
        image = ax.imshow(
            data,
            extent=extent,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("longitude" if longitude is not None else "column")
        ax.set_ylabel("latitude" if latitude is not None else "row")
        _add_scale_bar(ax, extent, latitude)
        fig.colorbar(image, ax=ax, shrink=0.82, label=label)

    fig.suptitle("Slope preservation and smoothing check", fontsize=14)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output

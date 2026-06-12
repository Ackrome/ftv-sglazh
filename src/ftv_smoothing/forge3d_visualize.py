"""Forge3D terrain renders for before/after DEM comparisons."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from .io import open_dataset

LOGGER = logging.getLogger(__name__)

_TERRAIN_COLORS = (
    "#005a32",
    "#238443",
    "#78c679",
    "#d9ef8b",
    "#fee08b",
    "#d8b365",
    "#8c510a",
    "#f7f7f7",
)


@dataclass(frozen=True, slots=True)
class Forge3DConfig:
    """Rendering-only controls for the Forge3D terrain comparison."""

    width: int = 800
    height: int = 520
    max_side: int = 900
    resolution_m: float = 30.0
    vertical_exaggeration: float = 2.4
    camera_radius_factor: float = 1.15
    camera_phi_deg: float = 45.0
    camera_theta_deg: float = 45.0
    fov_y_deg: float = 55.0
    sun_azimuth_deg: float = 225.0
    sun_elevation_deg: float = 28.0
    sun_intensity: float = 4.0
    rotation_quarter_turns_clockwise: int = 1

    def validate(self) -> None:
        """Reject values that cannot produce a meaningful render."""

        if self.width < 64 or self.height < 64:
            raise ValueError("Forge3D frame dimensions must be at least 64 pixels")
        if self.max_side < 64:
            raise ValueError("Forge3D max_side must be at least 64 pixels")
        if self.resolution_m <= 0:
            raise ValueError("Forge3D resolution_m must be positive")
        if self.vertical_exaggeration <= 0:
            raise ValueError("Forge3D vertical_exaggeration must be positive")
        if self.camera_radius_factor <= 0:
            raise ValueError("Forge3D camera_radius_factor must be positive")
        if self.rotation_quarter_turns_clockwise not in {0, 1, 2, 3}:
            raise ValueError("Forge3D rotation must be 0, 90, 180, or 270 degrees")


@dataclass(frozen=True, slots=True)
class PreparedHeightmaps:
    """A paired, rendering-only DEM representation with shared scaling."""

    before: np.ndarray
    after: np.ndarray
    valid_mask: np.ndarray
    domain: tuple[float, float]
    floor_elevation: float
    terrain_span_m: float
    source_shape: tuple[int, int]
    render_shape: tuple[int, int]
    valid_fraction: float


@dataclass(frozen=True, slots=True)
class Forge3DArtifacts:
    """Paths and render metadata emitted by a Forge3D comparison."""

    comparison_png: Path
    before_png: Path | None
    after_png: Path | None
    metadata_json: Path
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Forge3DTurntableArtifacts:
    """Paths emitted for an interactive Forge3D turntable viewer."""

    output_dir: Path
    manifest_json: Path
    animation_webp: Path
    metadata: dict[str, Any]


@dataclass(slots=True)
class _Forge3DContext:
    f3d: Any
    make_terrain_params_config: Any
    renderer: Any
    materials: Any
    ibl: Any
    overlays: list[Any]


def parse_crop(value: str) -> tuple[slice, slice]:
    """Parse a NumPy-style row and column crop."""

    try:
        row_text, col_text = value.split(",")
        row_start, row_stop = (int(part) for part in row_text.split(":"))
        col_start, col_stop = (int(part) for part in col_text.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "crop must use row_start:row_stop,col_start:col_stop"
        ) from exc
    return slice(row_start, row_stop), slice(col_start, col_stop)


def _valid_extent(mask: np.ndarray, padding: int = 2) -> tuple[slice, slice]:
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        raise ValueError("Forge3D rendering requires at least one finite paired DEM cell")
    row_start = max(0, int(rows[0]) - padding)
    row_stop = min(mask.shape[0], int(rows[-1]) + padding + 1)
    col_start = max(0, int(columns[0]) - padding)
    col_stop = min(mask.shape[1], int(columns[-1]) + padding + 1)
    return slice(row_start, row_stop), slice(col_start, col_stop)


def _resize_heightmap(
    array: np.ndarray,
    mask: np.ndarray,
    floor_elevation: float,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    scale = min(1.0, max_side / float(max(array.shape)))
    if scale == 1.0:
        return array.copy(), mask.copy()
    output_shape = tuple(max(2, int(round(size * scale))) for size in array.shape)
    zoom = tuple(output / source for output, source in zip(output_shape, array.shape))
    resized = ndi.zoom(array, zoom, order=1, prefilter=False)
    resized_mask = ndi.zoom(mask.astype(np.uint8), zoom, order=0, prefilter=False).astype(bool)
    resized[~resized_mask] = floor_elevation
    return np.ascontiguousarray(resized, dtype=np.float32), resized_mask


def prepare_heightmaps(
    before: np.ndarray,
    after: np.ndarray,
    *,
    resolution_m: float,
    max_side: int,
    crop_valid_extent: bool = True,
    rotation_quarter_turns_clockwise: int = 1,
) -> PreparedHeightmaps:
    """Prepare a shared-scale copy of two DEMs for visualization only."""

    original = np.asarray(before, dtype=np.float32)
    corrected = np.asarray(after, dtype=np.float32)
    if original.ndim != 2 or corrected.ndim != 2:
        raise ValueError("Forge3D DEM inputs must be two-dimensional")
    if original.shape != corrected.shape:
        raise ValueError("Forge3D before and after DEM shapes must match")
    if resolution_m <= 0:
        raise ValueError("Forge3D resolution_m must be positive")
    if max_side < 2:
        raise ValueError("Forge3D max_side must be at least 2 pixels")
    if rotation_quarter_turns_clockwise not in {0, 1, 2, 3}:
        raise ValueError("Forge3D rotation must be 0, 90, 180, or 270 degrees")

    valid = np.isfinite(original) & np.isfinite(corrected)
    if crop_valid_extent:
        extent = _valid_extent(valid)
        original = original[extent]
        corrected = corrected[extent]
        valid = valid[extent]
    if not valid.any():
        raise ValueError("Forge3D rendering requires at least one finite paired DEM cell")

    valid_before = original[valid]
    valid_after = corrected[valid]
    minimum = float(min(np.min(valid_before), np.min(valid_after)))
    maximum = float(max(np.max(valid_before), np.max(valid_after)))
    amplitude = max(maximum - minimum, 1.0)
    floor_elevation = minimum - max(10.0, 0.04 * amplitude)
    filled_before = np.where(valid, original, floor_elevation).astype(np.float32)
    filled_after = np.where(valid, corrected, floor_elevation).astype(np.float32)
    if rotation_quarter_turns_clockwise:
        rotation = -rotation_quarter_turns_clockwise
        filled_before = np.rot90(filled_before, k=rotation).copy()
        filled_after = np.rot90(filled_after, k=rotation).copy()
        valid = np.rot90(valid, k=rotation).copy()
    source_shape = filled_before.shape
    render_before, render_valid = _resize_heightmap(
        filled_before,
        valid,
        floor_elevation,
        max_side,
    )
    render_after, _ = _resize_heightmap(
        filled_after,
        valid,
        floor_elevation,
        max_side,
    )
    terrain_span_m = float(max(source_shape) * resolution_m)
    return PreparedHeightmaps(
        before=render_before,
        after=render_after,
        valid_mask=render_valid,
        domain=(minimum, maximum),
        floor_elevation=floor_elevation,
        terrain_span_m=terrain_span_m,
        source_shape=source_shape,
        render_shape=render_before.shape,
        valid_fraction=float(valid.mean()),
    )


def _ensure_neutral_hdr(cache_dir: Path) -> Path:
    """Write the minimal neutral RGBE environment required by Forge3D."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "neutral_environment.hdr"
    if path.exists():
        return path
    header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n"
    # Old-style RGBE scanlines are valid for widths below eight pixels.
    pixels = bytes([144, 157, 181, 129]) * 8
    path.write_bytes(header + pixels)
    return path


def _terrain_colormap(f3d: Any, domain: tuple[float, float]) -> Any:
    minimum, maximum = domain
    stops = [
        (minimum + index * (maximum - minimum) / (len(_TERRAIN_COLORS) - 1), color)
        for index, color in enumerate(_TERRAIN_COLORS)
    ]
    return f3d.Colormap1D.from_stops(stops=stops, domain=domain)


def _build_render_context(
    prepared: PreparedHeightmaps,
    output_dir: Path,
) -> _Forge3DContext:
    try:
        import forge3d as f3d
        from forge3d.terrain_params import make_terrain_params_config
    except ImportError as exc:
        raise RuntimeError(
            "Forge3D is required for 3D output; install forge3d==1.26.0"
        ) from exc

    hdr_path = _ensure_neutral_hdr(output_dir / ".forge3d-cache")
    colormap = _terrain_colormap(f3d, prepared.domain)
    overlays = [
        f3d.OverlayLayer.from_colormap1d(
            colormap,
            strength=1.0,
            offset=0.0,
            blend_mode="Alpha",
            domain=prepared.domain,
        )
    ]
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    materials = f3d.MaterialSet.terrain_default(
        triplanar_scale=6.0,
        normal_strength=1.0,
        blend_sharpness=4.0,
    )
    ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0, quality="low")
    ibl.set_base_resolution(64)
    return _Forge3DContext(
        f3d=f3d,
        make_terrain_params_config=make_terrain_params_config,
        renderer=renderer,
        materials=materials,
        ibl=ibl,
        overlays=overlays,
    )


def _render_params(
    prepared: PreparedHeightmaps,
    config: Forge3DConfig,
    context: _Forge3DContext,
    *,
    camera_phi_deg: float | None = None,
) -> Any:
    terrain_span = prepared.terrain_span_m
    if camera_phi_deg is None:
        camera_phi_deg = config.camera_phi_deg
    native_config = context.make_terrain_params_config(
        size_px=(config.width, config.height),
        render_scale=1.0,
        terrain_span=terrain_span,
        msaa_samples=1,
        z_scale=config.vertical_exaggeration,
        exposure=1.0,
        domain=prepared.domain,
        albedo_mode="mix",
        colormap_strength=0.82,
        ibl_enabled=False,
        light_azimuth_deg=config.sun_azimuth_deg,
        light_elevation_deg=config.sun_elevation_deg,
        sun_intensity=config.sun_intensity,
        ibl_intensity=0.0,
        cam_radius=terrain_span * config.camera_radius_factor,
        cam_phi_deg=camera_phi_deg,
        cam_theta_deg=config.camera_theta_deg,
        fov_y_deg=config.fov_y_deg,
        camera_mode="mesh",
        clip=(0.1, max(6000.0, terrain_span * 3.0)),
        overlays=context.overlays,
    )
    return context.f3d.TerrainRenderParams(native_config)


def _render_heightmap(
    array: np.ndarray,
    context: _Forge3DContext,
    render_params: Any,
) -> np.ndarray:
    frame = context.renderer.render_terrain_pbr_pom(
        material_set=context.materials,
        env_maps=context.ibl,
        params=render_params,
        heightmap=np.flipud(array).copy(),
        target=None,
    )
    return np.asarray(frame.to_numpy()).copy()


def _render_metadata(
    prepared: PreparedHeightmaps,
    config: Forge3DConfig,
    context: _Forge3DContext,
) -> dict[str, Any]:
    device = context.f3d.device_probe()
    LOGGER.info("Forge3D device: %s", device)
    return {
        "device": device,
        "renderer": context.renderer.info(),
        "source_shape": prepared.source_shape,
        "render_shape": prepared.render_shape,
        "valid_fraction": prepared.valid_fraction,
        "domain_m": prepared.domain,
        "floor_elevation_m": prepared.floor_elevation,
        "terrain_span_m": prepared.terrain_span_m,
        "vertical_exaggeration": config.vertical_exaggeration,
        "rotation_clockwise_deg": config.rotation_quarter_turns_clockwise * 90,
    }


def _render_frames(
    prepared: PreparedHeightmaps,
    output_dir: Path,
    config: Forge3DConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    context = _build_render_context(prepared, output_dir)
    render_params = _render_params(prepared, config, context)
    before_frame = _render_heightmap(prepared.before, context, render_params)
    after_frame = _render_heightmap(prepared.after, context, render_params)
    return before_frame, after_frame, _render_metadata(prepared, config, context)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _save_comparison(
    before: np.ndarray,
    after: np.ndarray,
    output_png: Path,
    *,
    metadata: dict[str, Any],
    source_label: str,
    write_panels: bool,
) -> tuple[Path | None, Path | None]:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    before_image = Image.fromarray(before).convert("RGB")
    after_image = Image.fromarray(after).convert("RGB")
    if write_panels:
        before_png = output_png.with_name(f"{output_png.stem}_before{output_png.suffix}")
        after_png = output_png.with_name(f"{output_png.stem}_after{output_png.suffix}")
        before_image.save(before_png)
        after_image.save(after_png)
    else:
        before_png = None
        after_png = None

    margin = 20
    header_height = 72
    footer_height = 58
    gap = 16
    frame_width, frame_height = before_image.size
    canvas = Image.new(
        "RGB",
        (margin * 2 + frame_width * 2 + gap, header_height + frame_height + footer_height),
        color=(18, 21, 31),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(25)
    panel_font = _font(18)
    detail_font = _font(15)
    draw.text(
        (margin, 12),
        f"Forge3D terrain comparison: {source_label}",
        fill=(245, 247, 250),
        font=title_font,
    )
    draw.text((margin, 46), "Before FTV", fill=(245, 247, 250), font=panel_font)
    draw.text(
        (margin + frame_width + gap, 46),
        "After FTV",
        fill=(245, 247, 250),
        font=panel_font,
    )
    canvas.paste(before_image, (margin, header_height))
    canvas.paste(after_image, (margin + frame_width + gap, header_height))
    detail = (
        f"Render grid {metadata['render_shape'][0]}x{metadata['render_shape'][1]} | "
        f"terrain span {metadata['terrain_span_m'] / 1000.0:.1f} km | "
        f"vertical exaggeration {metadata['vertical_exaggeration']:.2f}x | "
        f"valid cells {metadata['valid_fraction'] * 100.0:.1f}%"
    )
    draw.text(
        (margin, header_height + frame_height + 18),
        detail,
        fill=(205, 211, 220),
        font=detail_font,
    )
    canvas.save(output_png)
    return before_png, after_png


def render_forge3d_comparison(
    before: np.ndarray,
    after: np.ndarray,
    output_png: str | Path,
    *,
    config: Forge3DConfig | None = None,
    source_label: str = "DEM",
    write_panels: bool = True,
) -> Forge3DArtifacts:
    """Render paired Forge3D terrain images and a side-by-side comparison."""

    if config is None:
        config = Forge3DConfig()
    config.validate()
    output = Path(output_png)
    prepared = prepare_heightmaps(
        before,
        after,
        resolution_m=config.resolution_m,
        max_side=config.max_side,
        rotation_quarter_turns_clockwise=config.rotation_quarter_turns_clockwise,
    )
    before_frame, after_frame, metadata = _render_frames(prepared, output.parent, config)
    before_png, after_png = _save_comparison(
        before_frame,
        after_frame,
        output,
        metadata=metadata,
        source_label=source_label,
        write_panels=write_panels,
    )
    metadata_json = output.with_suffix(".json")
    metadata_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    LOGGER.info("Wrote Forge3D comparison to %s", output)
    return Forge3DArtifacts(output, before_png, after_png, metadata_json, metadata)


def _split_frame(before: np.ndarray, after: np.ndarray) -> Image.Image:
    before_image = Image.fromarray(before).convert("RGB")
    after_image = Image.fromarray(after).convert("RGB")
    width, height = before_image.size
    divider = width // 2
    combined = before_image.copy()
    combined.paste(after_image.crop((divider, 0, width, height)), (divider, 0))
    draw = ImageDraw.Draw(combined)
    draw.line((divider, 0, divider, height), fill=(247, 179, 74), width=3)
    return combined


def render_forge3d_turntable(
    before: np.ndarray,
    after: np.ndarray,
    output_dir: str | Path,
    *,
    config: Forge3DConfig | None = None,
    frame_count: int = 24,
    source_label: str = "DEM",
    scene_label: str = "ROI detail",
) -> Forge3DTurntableArtifacts:
    """Render a rotating before/after terrain stand as a report animation."""

    if config is None:
        config = Forge3DConfig()
    config.validate()
    if frame_count < 4:
        raise ValueError("Forge3D turntable needs at least four frames")
    output = Path(output_dir)
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_heightmaps(
        before,
        after,
        resolution_m=config.resolution_m,
        max_side=config.max_side,
        rotation_quarter_turns_clockwise=config.rotation_quarter_turns_clockwise,
    )
    context = _build_render_context(prepared, output)
    metadata = _render_metadata(prepared, config, context)
    metadata["frame_count"] = frame_count
    frames: list[dict[str, Any]] = []
    animated_frames: list[Image.Image] = []
    for index in range(frame_count):
        azimuth = (config.camera_phi_deg + index * 360.0 / frame_count) % 360.0
        LOGGER.info(
            "Rendering Forge3D turntable frame %d/%d at azimuth %.1f deg",
            index + 1,
            frame_count,
            azimuth,
        )
        render_params = _render_params(
            prepared,
            config,
            context,
            camera_phi_deg=azimuth,
        )
        before_frame = _render_heightmap(prepared.before, context, render_params)
        after_frame = _render_heightmap(prepared.after, context, render_params)
        before_name = f"frames/before_{index:03d}.png"
        after_name = f"frames/after_{index:03d}.png"
        Image.fromarray(before_frame).save(output / before_name)
        Image.fromarray(after_frame).save(output / after_name)
        animated_frames.append(_split_frame(before_frame, after_frame))
        frames.append(
            {
                "index": index,
                "azimuth_deg": round(azimuth, 3),
                "before": before_name,
                "after": after_name,
            }
        )
    animation_webp = output / "turntable_comparison.webp"
    animated_frames[0].save(
        animation_webp,
        save_all=True,
        append_images=animated_frames[1:],
        duration=125,
        loop=0,
        format="WEBP",
        quality=84,
        method=4,
    )
    manifest = {
        "schema_version": 1,
        "title": "FTV Terrain Viewer",
        "source": source_label,
        "scene": scene_label,
        "metadata": metadata,
        "frames": frames,
    }
    manifest_json = output / "manifest.json"
    manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    LOGGER.info("Wrote Forge3D turntable animation to %s", animation_webp)
    return Forge3DTurntableArtifacts(
        output_dir=output,
        manifest_json=manifest_json,
        animation_webp=animation_webp,
        metadata=metadata,
    )


def render_dataset_comparison(
    input_nc: str | Path,
    output_png: str | Path,
    *,
    before_var: str = "reprojected_dem",
    after_var: str = "reprojected_dem_ftv",
    crop: tuple[slice, slice] | None = None,
    config: Forge3DConfig | None = None,
    write_panels: bool = True,
) -> Forge3DArtifacts:
    """Render a comparison directly from an existing FTV NetCDF artifact."""

    path = Path(input_nc)
    with open_dataset(path) as ds:
        if before_var not in ds or after_var not in ds:
            raise KeyError(f"NetCDF must contain {before_var!r} and {after_var!r}")
        before = ds[before_var]
        after = ds[after_var]
        if crop is not None:
            row_slice, col_slice = crop
            before = before.isel(latitude=row_slice, longitude=col_slice)
            after = after.isel(latitude=row_slice, longitude=col_slice)
        if config is None:
            config = Forge3DConfig(resolution_m=float(ds.attrs.get("resolution_m", 30.0)))
        return render_forge3d_comparison(
            before.values,
            after.values,
            output_png,
            config=config,
            source_label=path.name,
            write_panels=write_panels,
        )


def render_dataset_turntable(
    input_nc: str | Path,
    output_dir: str | Path,
    *,
    before_var: str = "reprojected_dem",
    after_var: str = "reprojected_dem_ftv",
    crop: tuple[slice, slice] | None = None,
    config: Forge3DConfig | None = None,
    frame_count: int = 24,
    scene_label: str = "ROI detail",
) -> Forge3DTurntableArtifacts:
    """Render a turntable viewer directly from an existing FTV NetCDF."""

    path = Path(input_nc)
    with open_dataset(path) as ds:
        if before_var not in ds or after_var not in ds:
            raise KeyError(f"NetCDF must contain {before_var!r} and {after_var!r}")
        before = ds[before_var]
        after = ds[after_var]
        if crop is not None:
            row_slice, col_slice = crop
            before = before.isel(latitude=row_slice, longitude=col_slice)
            after = after.isel(latitude=row_slice, longitude=col_slice)
        if config is None:
            config = Forge3DConfig(resolution_m=float(ds.attrs.get("resolution_m", 30.0)))
        return render_forge3d_turntable(
            before.values,
            after.values,
            output_dir,
            config=config,
            frame_count=frame_count,
            source_label=path.name,
            scene_label=scene_label,
        )


def build_parser() -> argparse.ArgumentParser:
    """Create the standalone Forge3D CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_nc", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--before-var", default="reprojected_dem")
    parser.add_argument("--after-var", default="reprojected_dem_ftv")
    parser.add_argument("--crop", type=parse_crop)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--max-side", type=int, default=900)
    parser.add_argument("--resolution-m", type=float)
    parser.add_argument("--vertical-exaggeration", type=float, default=2.4)
    parser.add_argument(
        "--rotation-clockwise",
        type=int,
        choices=(0, 90, 180, 270),
        default=90,
    )
    parser.add_argument("--skip-panels", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Render Forge3D before/after terrain images from an existing NetCDF."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.resolution_m is None:
        with open_dataset(args.input_nc) as ds:
            resolution_m = float(ds.attrs.get("resolution_m", 30.0))
    else:
        resolution_m = args.resolution_m
    config = Forge3DConfig(
        width=args.width,
        height=args.height,
        max_side=args.max_side,
        resolution_m=resolution_m,
        vertical_exaggeration=args.vertical_exaggeration,
        rotation_quarter_turns_clockwise=args.rotation_clockwise // 90,
    )
    render_dataset_comparison(
        args.input_nc,
        args.output,
        before_var=args.before_var,
        after_var=args.after_var,
        crop=args.crop,
        config=config,
        write_panels=not args.skip_panels,
    )


def build_turntable_parser() -> argparse.ArgumentParser:
    """Create the Forge3D turntable viewer CLI parser."""

    parser = argparse.ArgumentParser(description=render_forge3d_turntable.__doc__)
    parser.add_argument("input_nc", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--before-var", default="reprojected_dem")
    parser.add_argument("--after-var", default="reprojected_dem_ftv")
    parser.add_argument("--crop", type=parse_crop)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--scene-label", default="ROI detail")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--max-side", type=int, default=900)
    parser.add_argument("--resolution-m", type=float)
    parser.add_argument("--vertical-exaggeration", type=float, default=2.4)
    parser.add_argument(
        "--rotation-clockwise",
        type=int,
        choices=(0, 90, 180, 270),
        default=90,
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def turntable_main(argv: list[str] | None = None) -> None:
    """Render a Forge3D turntable report animation."""

    args = build_turntable_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.resolution_m is None:
        with open_dataset(args.input_nc) as ds:
            resolution_m = float(ds.attrs.get("resolution_m", 30.0))
    else:
        resolution_m = args.resolution_m
    config = Forge3DConfig(
        width=args.width,
        height=args.height,
        max_side=args.max_side,
        resolution_m=resolution_m,
        vertical_exaggeration=args.vertical_exaggeration,
        rotation_quarter_turns_clockwise=args.rotation_clockwise // 90,
    )
    render_dataset_turntable(
        args.input_nc,
        args.output_dir,
        before_var=args.before_var,
        after_var=args.after_var,
        crop=args.crop,
        config=config,
        frame_count=args.frames,
        scene_label=args.scene_label,
    )


if __name__ == "__main__":
    main()

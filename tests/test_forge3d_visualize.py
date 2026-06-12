import numpy as np

import ftv_smoothing.forge3d_visualize as forge3d_visualize
from ftv_smoothing.forge3d_visualize import (
    parse_crop,
    prepare_heightmaps,
)
from ftv_smoothing.webgl_export import (
    _copy_viewer_assets,
    _downsample_full_extent,
    export_webgl_model,
)


def test_prepare_heightmaps_preserves_shared_scale_and_mask() -> None:
    before = np.arange(80 * 100, dtype=np.float32).reshape(80, 100)
    after = before + 2.0
    before[:5, :] = np.nan
    after[:5, :] = np.nan
    before[30:35, 40:45] = np.nan
    after[30:35, 40:45] = np.nan

    prepared = prepare_heightmaps(
        before,
        after,
        resolution_m=30.0,
        max_side=50,
    )

    assert prepared.source_shape == (100, 77)
    assert prepared.render_shape == (50, 38)
    assert prepared.domain == (500.0, 8001.0)
    assert prepared.terrain_span_m == 3000.0
    assert prepared.before.dtype == np.float32
    assert prepared.after.dtype == np.float32
    assert np.all(prepared.before[~prepared.valid_mask] == prepared.floor_elevation)
    assert np.all(prepared.after[~prepared.valid_mask] == prepared.floor_elevation)


def test_prepare_heightmaps_rejects_unpaired_shapes() -> None:
    before = np.ones((8, 9), dtype=np.float32)
    after = np.ones((8, 8), dtype=np.float32)

    try:
        prepare_heightmaps(before, after, resolution_m=30.0, max_side=64)
    except ValueError as exc:
        assert "shapes must match" in str(exc)
    else:
        raise AssertionError("shape mismatch must be rejected")


def test_parse_crop_uses_numpy_order() -> None:
    rows, columns = parse_crop("12:34,56:78")

    assert rows == slice(12, 34)
    assert columns == slice(56, 78)


def test_prepare_heightmaps_can_disable_clockwise_rotation() -> None:
    before = np.arange(12, dtype=np.float32).reshape(3, 4)
    after = before + 1.0

    prepared = prepare_heightmaps(
        before,
        after,
        resolution_m=30.0,
        max_side=64,
        crop_valid_extent=False,
        rotation_quarter_turns_clockwise=0,
    )

    assert prepared.source_shape == (3, 4)
    assert np.array_equal(prepared.before, before)


def test_copy_viewer_assets_writes_static_application(tmp_path) -> None:
    viewer_html = _copy_viewer_assets(tmp_path)

    assert viewer_html == tmp_path / "index.html"
    assert "FTV Terrain Viewer" in viewer_html.read_text(encoding="utf-8")
    assert (tmp_path / "styles.css").exists()
    assert (tmp_path / "app.js").exists()


def test_downsample_webgl_model_keeps_complete_extent() -> None:
    before = np.arange(80 * 100, dtype=np.float32).reshape(80, 100)
    after = before + 1
    before[0, 0] = np.nan
    after[0, 0] = np.nan

    sampled_before, sampled_after, sampled_mask, sampled_segments = _downsample_full_extent(
        before,
        after,
        max_side=50,
    )

    assert sampled_before.shape == (40, 50)
    assert sampled_after.shape == (40, 50)
    assert sampled_mask.shape == (40, 50)
    assert sampled_segments.shape == (40, 50)
    assert sampled_segments.max() == 0
    assert sampled_mask[0, 0] == 0
    assert sampled_mask[-1, -1] == 255


def test_export_webgl_model_writes_binary_mesh_payload(tmp_path) -> None:
    before = np.arange(64 * 72, dtype=np.float32).reshape(64, 72)
    after = before + 2
    before[4:8, 10:13] = np.nan
    after[4:8, 10:13] = np.nan

    artifacts = export_webgl_model(before, after, tmp_path)

    assert artifacts.viewer_html.exists()
    assert artifacts.model_json.exists()
    assert artifacts.before_f32.stat().st_size == 64 * 72 * 4
    assert artifacts.after_f32.stat().st_size == 64 * 72 * 4
    assert artifacts.mask_u8.stat().st_size == 64 * 72
    assert artifacts.metadata["whole_study_area"] is True
    assert artifacts.metadata["base_surface"] == "horizontal XZ plane"


def test_comparison_cli_calls_dataset_renderer(monkeypatch, tmp_path) -> None:
    captured = {}

    def capture(*args, **kwargs) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(forge3d_visualize, "render_dataset_comparison", capture)
    forge3d_visualize.main(
        [
            "input.nc",
            "--output",
            str(tmp_path / "comparison.png"),
            "--resolution-m",
            "30",
        ]
    )

    assert captured["args"][0] == forge3d_visualize.Path("input.nc")
    assert captured["kwargs"]["config"].rotation_quarter_turns_clockwise == 1


def test_turntable_cli_calls_dataset_renderer(monkeypatch, tmp_path) -> None:
    captured = {}

    def capture(*args, **kwargs) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(forge3d_visualize, "render_dataset_turntable", capture)
    forge3d_visualize.turntable_main(
        [
            "input.nc",
            "--output-dir",
            str(tmp_path / "viewer"),
            "--resolution-m",
            "30",
            "--frames",
            "12",
        ]
    )

    assert captured["args"][0] == forge3d_visualize.Path("input.nc")
    assert captured["kwargs"]["frame_count"] == 12
    assert captured["kwargs"]["config"].rotation_quarter_turns_clockwise == 1

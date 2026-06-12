from pathlib import Path
from types import SimpleNamespace

import ftv_smoothing.cli as cli
import ftv_smoothing.worker as worker
from ftv_smoothing.app_core import normalize_request, parse_crop_text
from ftv_smoothing.job_store import JobStore


def test_parse_crop_text_uses_numpy_order() -> None:
    rows, columns = parse_crop_text("12:34,56:78")

    assert rows == slice(12, 34)
    assert columns == slice(56, 78)


def test_normalize_request_snaps_numeric_values(tmp_path: Path) -> None:
    input_nc = tmp_path / "input.nc"
    input_nc.write_text("synthetic", encoding="utf-8")

    params = normalize_request(
        {
            "input_nc": str(input_nc),
            "alpha": "1.53",
            "k_size": "10.6",
            "webgl_max_side": "513",
            "roi_sample_count": "12",
            "vertical_exaggeration": "2.96",
        },
        input_nc,
    )

    assert params["alpha"] == 1.55
    assert params["k_size"] == 11
    assert params["webgl_max_side"] == 512
    assert params["roi_sample_count"] == 4
    assert params["vertical_exaggeration"] == 3.0


def test_job_store_persists_progress(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3", tmp_path / "results")
    job = store.create_job(
        cache_key="0123456789abcdef",
        parameters={"alpha": 1.5},
        input_fingerprint={"path": "input.nc"},
    )

    store.update_progress(
        job["id"],
        status="running",
        progress_percent=42,
        stage="Optimizing FTV",
    )
    updated = store.get_job(job["id"])

    assert updated is not None
    assert updated["status"] == "running"
    assert updated["progress_percent"] == 42
    assert updated["stage"] == "Optimizing FTV"


def test_worker_executes_job_and_reuses_completed_cache(monkeypatch, tmp_path: Path) -> None:
    input_nc = tmp_path / "input.nc"
    input_nc.write_text("synthetic", encoding="utf-8")
    store = JobStore(tmp_path / "jobs.sqlite3", tmp_path / "results")
    calls = []
    params = {
        "input_nc": str(input_nc),
        "crop": "1:4,2:5",
        "alpha": 1.4,
        "k_size": 10,
        "lambda_base": 0.5,
        "max_iter": 3,
        "tol": 1e-4,
        "backend": "cpu",
        "convolution_method": "direct",
        "msa_threshold": 8.5,
        "interpolation_iterations": 2,
        "visualization_dpi": 180,
        "save_sarp": False,
        "postprocess": False,
        "webgl_max_side": 512,
        "vertical_exaggeration": 2.4,
        "roi_sample_count": 2,
    }
    fingerprint = {"path": str(input_nc), "size_bytes": input_nc.stat().st_size, "mtime_ns": 1}

    def fake_orchestrate_file(
        input_path,
        output_nc,
        output_png,
        *,
        diagnostics_png=None,
        slope_comparison_png=None,
        forge3d_png=None,
        snapshot_dir=None,
        config=None,
        crop=None,
        progress_callback=None,
    ):
        calls.append((input_path, config.msa_threshold, crop))
        if progress_callback is not None:
            progress_callback("Optimizing FTV", 50, {"iteration": 1, "max_iter": 3})
        Path(output_nc).write_text("netcdf", encoding="utf-8")
        Path(output_png).write_text("png", encoding="utf-8")
        Path(diagnostics_png).write_text("diagnostics", encoding="utf-8")
        Path(slope_comparison_png).write_text("slope", encoding="utf-8")
        Path(output_nc).with_suffix(".metrics.json").write_text(
            '{"optimizer_iterations": 3, "artifact_fraction_of_valid": 0.1, "slope_rmse_before_after_deg": 0.2}',
            encoding="utf-8",
        )

    def fake_export_dataset_webgl_model(input_path, output_dir, *, config=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "index.html").write_text("<html></html>", encoding="utf-8")
        Path(output_dir, "terrain_before.f32").write_bytes(b"1234")
        Path(output_dir, "terrain_after.f32").write_bytes(b"1234")
        Path(output_dir, "terrain_mask.u8").write_bytes(b"1")
        Path(output_dir, "terrain-model.json").write_text(
            '{"files": {"before": "terrain_before.f32", "after": "terrain_after.f32", "mask": "terrain_mask.u8"}}',
            encoding="utf-8",
        )
        return SimpleNamespace(metadata={"grid_shape": [2, 2]})

    def fake_export_changed_roi_samples(input_path, output_dir, *, count=0, config=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "index.json").write_text('{"samples": []}', encoding="utf-8")
        return {"schema_version": 1, "sample_count": count, "samples": []}

    monkeypatch.setattr(worker, "orchestrate_file", fake_orchestrate_file)
    monkeypatch.setattr(worker, "export_dataset_webgl_model", fake_export_dataset_webgl_model)
    monkeypatch.setattr(worker, "export_changed_roi_samples", fake_export_changed_roi_samples)

    first = store.create_job(
        cache_key="0123456789abcdef",
        parameters=params,
        input_fingerprint=fingerprint,
    )
    first_result = worker.execute_job(first["id"], store=store)
    second = store.create_job(
        cache_key="0123456789abcdef",
        parameters=params,
        input_fingerprint=fingerprint,
    )
    second_result = worker.execute_job(second["id"], store=store)
    completed = store.public_job(store.get_job(second["id"]))

    assert first_result["cache_hit"] is False
    assert second_result["cache_hit"] is True
    assert calls == [(str(input_nc), 8.5, (slice(1, 4), slice(2, 5)))]
    assert completed["status"] == "completed"
    assert completed["progress_percent"] == 100
    assert "comparison_png" in completed["result"]["urls"]
    assert "webgl" in completed["result"]["urls"]
    assert completed["result"]["roi_samples"]["sample_count"] == 2


def test_cli_passes_msa_threshold_to_config(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_orchestrate_file(*args, **kwargs) -> None:
        captured["config"] = kwargs["config"]

    monkeypatch.setattr(cli, "orchestrate_file", fake_orchestrate_file)

    cli.main(
        [
            "input.nc",
            "--output",
            str(tmp_path / "output.nc"),
            "--max-iter",
            "1",
            "--msa-threshold",
            "9.5",
        ]
    )

    assert captured["config"].msa_threshold == 9.5

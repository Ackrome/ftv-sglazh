# FTV smoothing for masked DEM data

This project implements fractional-order total variation smoothing for the
`reprojected_dem` layer in `relief_masked.nc`. The pipeline includes:

- NaN-safe preprocessing with diffusion interpolation;
- Maximum Slope Approach (MSA) artifact detection;
- spatially adaptive fidelity weights (SARP) from `slope` and
  `max_curvature`;
- Grunwald-Letnikov fractional derivatives with a short-memory window;
- a Chambolle-Pock primal-dual optimizer;
- CuPy GPU acceleration and a Numba CPU fallback;
- iterative morphological filtering (IMF), validation metrics, and PNG
  comparisons;
- optional paired 3D terrain renders through
  [forge3d](https://github.com/milos-agathon/forge3d).

Install and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m ftv_smoothing.cli relief_masked.nc `
  --output artifacts\relief_masked_ftv.nc `
  --png artifacts\relief_masked_ftv_comparison.png `
  --diagnostics-png artifacts\relief_masked_ftv_diagnostics.png `
  --slope-comparison-png artifacts\relief_masked_ftv_slope_comparison.png `
  --forge3d-png artifacts\relief_masked_ftv_3d_comparison.png `
  --snapshot-dir artifacts\snapshots `
  --lambda-base 2.0
```

The output NetCDF preserves the original layers and adds
`reprojected_dem_ftv` and `ftv_artifact_mask`.

Run the browser application with Docker Compose:

```powershell
docker compose up --build
```

Open `http://localhost:8765`. Compose starts a FastAPI web/API service, a
Celery CPU worker, Redis, and Flower at `http://localhost:5555` for queue
inspection. The app exposes presets plus fields for the main FTV
hyperparameters (`alpha`,
`k_size`, `lambda_base`, `max_iter`, `tol`, `msa_threshold`, backend,
convolution method, crop, preview DPI, WebGL mesh size, vertical exaggeration,
postprocessing, and SARP saving). Jobs are queued in Celery, status and
progress are stored in SQLite at `artifacts\app-results\jobs.sqlite3`, and
generated NetCDF/PNG/WebGL artifacts are cached under `artifacts\app-results`.
Repeating the same input file and parameter set loads the saved result without
rerunning the optimizer. The UI can cancel, retry, and delete terminal jobs,
shows result storage usage, and can clean orphaned cached result folders.

Completed browser jobs embed the WebGL terrain viewer directly in the main
result panel instead of opening a separate tab. The worker also exports up to
four most-changed ROI samples: each row shows 2D before/after PNG crops with
highlighted high-change segments plus synchronized interactive 3D before/after
crops. The same high-change segment mask is shown as an amber overlay in the
3D crops. The WebUI supports English/Russian switching and snaps typed numeric
fields to the nearest allowed step before submission; FastAPI repeats this
normalization on the server.

New runs also generate `slope_comparison.png` and `validation_report.md`.
The slope report compares derived slope before/after and the slope delta, then
records slope RMSE, P95 absolute slope delta, and slope correlation metrics.

For NVIDIA GPU probing inside Docker, run with the override file:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The WebUI shows the selected backend, CuPy/CUDA versions, visible CUDA devices,
and the reason for GPU use or CPU fallback. `Backend = Auto` selects the usable
CUDA device with the most free VRAM and routes the job to the `gpu` Celery
queue. The default stack keeps a CPU worker on the `cpu` queue; the GPU override
adds `worker-gpu`. `Backend = GPU` fails fast if no usable GPU is available.
Legacy Kepler cards such as GTX 780 are expected to fall back to CPU with the
current CUDA 12+/13 CuPy stack.

Render 3D comparisons later from an existing FTV NetCDF without rerunning the
optimizer:

```powershell
.\.venv\Scripts\python.exe -m ftv_smoothing.forge3d_visualize `
  artifacts\relief_masked_ftv.nc `
  --output artifacts\relief_masked_ftv_3d_comparison.png
```

Forge3D report renders are rotated `90` degrees clockwise by default. Create
an optional ROI turntable animation:

```powershell
.\.venv\Scripts\ftv-render-turntable.exe artifacts\full\relief_masked_ftv.nc `
  --crop 2011:2611,1631:2331 `
  --output-dir artifacts\full\forge3d-roi-viewer
```

Export the browser-native full-area terrain model and serve it locally:

```powershell
.\.venv\Scripts\ftv-export-webgl.exe artifacts\full\relief_masked_ftv.nc `
  --output-dir artifacts\full\webgl-full-viewer `
  --max-side 640 `
  --vertical-exaggeration 2.4

.\.venv\Scripts\ftv-serve-3d.exe artifacts\full\webgl-full-viewer
```

Open `http://127.0.0.1:8765`. The WebGL viewer uses a live terrain mesh on a
horizontal XZ plane, keeps the complete study extent, and supports free orbit,
pan, zoom, auto rotation, vertical-scale adjustment, and live before/after,
blend, and correction-map layers.

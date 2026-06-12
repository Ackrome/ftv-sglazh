# FTV smoothing usage

## Docker Compose browser application

Start the application from the project root:

```powershell
docker compose up --build
```

Open `http://localhost:8765`. The Compose stack builds the Python package and
starts:

- `api`: FastAPI web/API service on port `8765`;
- `worker`: Celery worker that listens to the `cpu` queue and runs FTV
  calculations outside the web request;
- `redis`: Celery broker/result backend;
- `flower`: Celery queue monitor on `http://localhost:5555`.

To build the image with CuPy and expose NVIDIA devices to the API and worker
containers, use the GPU override:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

This requires the NVIDIA driver and NVIDIA Container Toolkit on the host. The
default `docker compose up --build` path does not install the GPU extra, so it
remains suitable for CPU-only machines.
The GPU override keeps the CPU worker and adds `worker-gpu`, which listens to
the `gpu` queue. FastAPI routes a job to `gpu` only when backend diagnostics
select a usable CUDA device; otherwise it routes to `cpu`.

The default input is `/data/relief_masked.nc`, mapped from the project root.
Persistent job state is stored in `artifacts\app-results\jobs.sqlite3`, and
generated NetCDF/PNG/WebGL artifacts are stored under
`artifacts\app-results`.
Docker builds use `constraints.txt` for top-level dependency and build-tool
versions, and the Dockerfile installs the editable package with
`--no-build-isolation` after installing the constrained build tooling.

The form fields control the run hyperparameters. Each field in the WebUI has an
`i` tooltip with the same meaning:

| Field | What it controls |
| --- | --- |
| `Input NetCDF` | Source dataset path inside the container. With the provided Compose file this is `/data/relief_masked.nc`. |
| `Crop` | Optional NumPy-order subset `row_start:row_stop,col_start:col_stop`. Empty value processes the full grid. |
| `Preset` | Fills the calculation fields with a named profile: fast preview, balanced, conservative, high quality, or aggressive smoothing. `Custom` keeps the current values. |
| `Alpha` | Fractional derivative order in `[1.0, 2.0]`. Lower values smooth more; higher values preserve sharper relief structures. |
| `K size` | Short-memory window for Grunwald-Letnikov fractional derivatives. Larger windows can improve long-range behavior but cost more time and memory. |
| `Lambda` | Base data-fidelity weight. Lower values allow stronger smoothing; higher values keep the output closer to the input DEM. |
| `Max iter` | Maximum Chambolle-Pock optimizer iterations. More iterations can improve convergence but increase runtime. |
| `Tolerance` | Early-stop threshold for relative optimizer change. Smaller values demand tighter convergence. |
| `MSA threshold` | Local slope anomaly threshold for artifact detection. Higher values mark fewer cells as artifacts. |
| `Backend` | Compute backend. `Auto` probes GPU first and falls back to CPU; `GPU` is strict and fails if no usable CUDA device exists. |
| `Convolution` | Fractional derivative implementation. `Direct` is usually best for short windows; `FFT` can help with large windows. |
| `Interpolation` | NaN-fill diffusion passes before optimization. More passes smooth larger missing regions before FTV starts. |
| `Preview DPI` | Resolution of generated PNG comparison and diagnostics previews. Higher values create larger files. |
| `WebGL side` | Maximum side length of the exported 3D terrain mesh. Higher values preserve more detail but increase viewer size. |
| `Vertical scale` | Vertical exaggeration for the WebGL terrain viewer. It affects visualization only, not the NetCDF result. The result panel also has a slider and a numeric field for changing this value interactively. |
| `ROI samples` | Number of most-changed areas to export as detailed 2D and 3D before/after examples. Use `0` to disable; the WebUI caps this at `4`. |
| `Postprocess` | Enables morphological cleanup after FTV to reduce isolated correction artifacts. |
| `Save SARP` | Stores the adaptive SARP fidelity field in the output NetCDF for later inspection. |

The diagnostics block above the form shows the selected compute backend,
CuPy/CUDA versions, visible CUDA devices, and the reason for GPU acceptance or
CPU fallback.

Number fields in the WebUI are snapped to the nearest valid value on blur,
change, and submit. FastAPI repeats the same snapping on the server with the
shared `/api/config` field specs, so API requests and browser submissions use
the same normalized parameters. This prevents browser-native step validation
popups such as `Enter a valid value` from blocking a run when the user types a
nearby value.

Use the language selector in the top bar to switch the interface between fully
English and Russian labels, controls, tooltips, and most runtime statuses. The
choice is stored in browser `localStorage`.

When `Queue or load cached` is pressed, FastAPI hashes the input file
fingerprint together with all selected fields. If a completed result with the
same key already exists in `artifacts\app-results`, the job completes at 100%
and the UI loads the saved comparison PNG, metrics, NetCDF, and WebGL viewer
link without rerunning the optimizer. Otherwise FastAPI creates a queued job
and Celery executes it in the worker container.

The WebUI subscribes to a server-sent events stream for the selected job and
shows the current stage and percentage. Progress is reported from pipeline
stages and optimizer iterations:

- preprocessing and MSA artifact detection;
- FTV optimizer iteration count;
- postprocessing;
- metrics and PNG rendering;
- slope comparison PNG rendering;
- exact-grid 1:1 PNG artifact export;
- NetCDF serialization;
- full-area WebGL model export;
- most-changed ROI sample export.

### Job management, storage, and cleanup

The result panel exposes actions for the selected job:

- `Cancel`: marks a queued/running job as cancelled, revokes the Celery task
  when possible, and asks the worker to stop through the SQLite job state;
- `Retry`: creates a new job from a failed or cancelled job's normalized
  parameters;
- `Delete`: removes a terminal job row from history. Result files are kept
  unless no remaining job references the same cache key and the API is called
  with `delete_files=true`;
- `Download artifacts`: appears after completion and downloads a ZIP archive
  with every file created in the selected run directory;
- `Cleanup`: removes orphaned result directories that are no longer referenced
  by any job row.

The history panel shows current storage usage from `/api/storage`: total bytes
under `artifacts\app-results` and the number of cached result folders. The
cleanup action refreshes this number after deleting orphaned folders.

The archive endpoint is `/api/jobs/{job_id}/artifacts.zip`. It builds the ZIP
from the cached result directory on demand and names the top-level ZIP folder
after the result cache key.

### 1:1 PNG artifacts

Each new completed run writes exact-grid PNGs under `pixel_1to1/`. These files
are intended for pixel-level inspection and external GIS/image tooling:

- `elevation_before_1to1.png`;
- `elevation_after_1to1.png`;
- `elevation_delta_1to1.png`;
- `slope_before_1to1.png`;
- `slope_after_1to1.png`;
- `slope_delta_1to1.png`;
- `index.json` with shape, color limits, resolution, and file names.

The mapping is exactly `1 DEM grid cell = 1 PNG pixel`; no DPI scaling,
resampling, figure padding, axes, or labels are applied. Invalid cells are
rendered as white pixels. These files are included in `Download artifacts`.

### Slope comparison and validation report

Each new web run writes `slope_comparison.png`. The image has three panels:

- slope before FTV;
- slope after FTV on the same color scale;
- slope delta `after - before`.

This is intended to answer the scientific question "did smoothing remove local
artifacts without destroying terrain gradients?". The metrics JSON now also
contains `slope_rmse_before_after_deg`, `slope_p95_abs_delta_deg`, and
`slope_correlation_before_after`. The result panel shows these metrics and a
download link to `validation_report.md`, which summarizes parameters,
elevation residuals, slope preservation, curvature variance, and a short
interpretation.

### Embedded 3D and changed-area table

Completed jobs show the 3D terrain viewer directly on the main page. The
viewer no longer opens a new browser tab from the result panel. It loads the
cached `webgl/terrain-model.json` and its binary heightfield files from the
selected result directory.

The `Most changed areas` section is generated from the absolute correction
`abs(reprojected_dem_ftv - reprojected_dem)`. The exporter ranks
non-overlapping windows by mean absolute correction, then writes up to four rows
per selected ROI. Each row has four visual columns:

| Column | Content |
| --- | --- |
| `2D before` | Topographic PNG crop before FTV. |
| `2D after` | Topographic PNG crop after FTV using the same color scale. |
| `3D before` | Interactive WebGL crop fixed to the source DEM layer. |
| `3D after` | Interactive WebGL crop fixed to the corrected DEM layer. |

Cells with the strongest changes inside each ROI are highlighted as red/yellow
segments on both 2D PNG crops. The same segment mask is exported as
`terrain_segments.u8` and shown as an amber overlay in the WebGL 3D crops. The
row metadata also stores the number and bounding boxes of these changed
segments.

All embedded 3D canvases share the same interaction settings. Dragging,
panning, zooming, or pressing `Fit` on the main 3D viewer synchronizes the
camera across the ROI before/after views. The shared `Vertical` slider and
numeric input apply the same vertical exaggeration to every embedded 3D view.
ROI WebGL views are loaded lazily and old offscreen contexts are released, so a
top-4 table stays inside the browser's WebGL context limit.

### Compute backend diagnostics and auto-selection

`Backend = Auto` probes CuPy and every visible CUDA device. A device is treated
as usable only after a small CuPy allocation, arithmetic operation, host
transfer, and stream synchronization succeed. If several devices pass, the app
selects the one with the most free VRAM, then uses total VRAM and compute
capability as tie-breakers. If no CUDA device passes, calculations run on the
Numba CPU backend.

`Backend = GPU` is strict. If no usable GPU is visible, FastAPI rejects a new
run with a clear error instead of silently running on CPU. Already saved
results can still be opened without a GPU because no recalculation is needed.

Kepler-era cards such as GTX 780 report compute capability `3.5`. With the
current CUDA 12+/13 CuPy wheels this architecture is normally not runnable. The
diagnostics panel will show the card as legacy and blocked if the runtime probe
fails, and `Auto` will select CPU.

## Full DEM processing

Run the complete pipeline against the source dataset:

```powershell
.\.venv\Scripts\python.exe -m ftv_smoothing.cli relief_masked.nc `
  --output artifacts\relief_masked_ftv.nc `
  --png artifacts\relief_masked_ftv_comparison.png `
  --diagnostics-png artifacts\relief_masked_ftv_diagnostics.png `
  --slope-comparison-png artifacts\relief_masked_ftv_slope_comparison.png `
  --forge3d-png artifacts\relief_masked_ftv_3d_comparison.png `
  --snapshot-dir artifacts\snapshots `
  --backend auto `
  --lambda-base 2.0
```

`--backend auto` uses the same CuPy probe and best-device selection as the
WebUI, then logs a warning before falling back to the Numba CPU backend when no
usable CUDA device is found. The resulting NetCDF keeps every source layer and
adds:

- `reprojected_dem_ftv`: corrected DEM with the original NaN topology;
- `ftv_artifact_mask`: MSA artifact mask stored as `int8`;
- `ftv_lambda_sarp`: optional SARP fidelity field when `--save-sarp` is used.

The command also writes JSON metrics next to the output NetCDF.

For the provided `relief_masked.nc`, a full-grid sweep compared
`lambda_base=1.0`, `1.5`, and `2.0`. The conservative `2.0` profile is
recommended: it preserved more curvature structure while reducing the RMS
elevation correction. The implementation default remains `1.0` for API
compatibility with the specification; pass `--lambda-base 2.0` for this
dataset.

## Intermediate visual comparisons

Use `--snapshot-dir artifacts\snapshots --snapshot-every 25` to save
before/current comparisons during optimization. Snapshots intentionally use a
lighter DPI than the final report to reduce runtime. The final
`--png` comparison is always rendered at the configured report DPI (600 by
default).

## Forge3D terrain comparisons

Use `--forge3d-png` during the main run to write a side-by-side 3D terrain
comparison and separate `*_before.png` and `*_after.png` panels. The renderer
uses the Vulkan-backed
[forge3d](https://github.com/milos-agathon/forge3d) `TerrainRenderer` with the
same metric scale and camera for both panels. Invalid exterior cells are shown
as a lowered base only in the render copy. NetCDF values are not changed.

To render an already processed NetCDF without rerunning the FTV optimizer:

```powershell
.\.venv\Scripts\python.exe -m ftv_smoothing.forge3d_visualize `
  artifacts\full\relief_masked_ftv.nc `
  --output artifacts\full\relief_masked_ftv_3d_comparison.png
```

For a close ROI inspection:

```powershell
.\.venv\Scripts\python.exe -m ftv_smoothing.forge3d_visualize `
  artifacts\full\relief_masked_ftv.nc `
  --crop 2011:2611,1631:2331 `
  --max-side 900 `
  --vertical-exaggeration 2.4 `
  --output artifacts\full\relief_masked_ftv_3d_roi_comparison.png
```

The standalone renderer also accepts `--width`, `--height`, `--resolution-m`,
`--before-var`, `--after-var`, `--rotation-clockwise`, and `--skip-panels`.
Forge3D render copies are rotated `90` degrees clockwise by default. Pass
`--rotation-clockwise 0` only when the original orientation is needed. A tiny neutral HDR
environment is generated in `.forge3d-cache` beside the output so rendering is
reproducible without downloading external assets.
Each comparison also writes a neighboring `.json` file with the Vulkan device,
render-grid size, metric span, valid-cell fraction, and vertical exaggeration.

## Forge3D turntable animation and web viewer

Generate a rotating ROI stand after the NetCDF result exists:

```powershell
.\.venv\Scripts\ftv-render-turntable.exe artifacts\full\relief_masked_ftv.nc `
  --crop 2011:2611,1631:2331 `
  --frames 24 `
  --max-side 900 `
  --vertical-exaggeration 2.4 `
  --output-dir artifacts\full\forge3d-roi-viewer
```

The output directory contains:

- `turntable_comparison.webp`: looping split-screen animation;
- `frames\before_*.png` and `frames\after_*.png`: synchronized Forge3D layers;
- `manifest.json`: azimuths and render metadata;
- `index.html`, `styles.css`, and `app.js`: self-contained static viewer.

Serve the generated viewer:

```powershell
.\.venv\Scripts\ftv-serve-3d.exe artifacts\full\forge3d-roi-viewer
```

Open `http://127.0.0.1:8765`. The viewer supports:

- horizontal drag, mouse wheel, arrow keys, and timeline scrubbing to rotate;
- play/pause animation;
- `Before FTV`, `After FTV`, and `Split` comparison modes;
- a draggable vertical before/after divider;
- render metadata and the active azimuth.

## Region-of-interest tuning

Before a long full-grid run, tune parameters on a crop:

```powershell
.\.venv\Scripts\python.exe -m ftv_smoothing.cli relief_masked.nc `
  --crop 900:1500,1500:2200 `
  --output artifacts\roi_ftv.nc `
  --png artifacts\roi_ftv_comparison.png `
  --diagnostics-png artifacts\roi_ftv_diagnostics.png `
  --max-iter 80
```

Crop coordinates use NumPy order: `row_start:row_stop,col_start:col_stop`.

## Main controls

- `--alpha`: fractional derivative order, normally `1.2` to `2.0`.
- `--k-size`: short-memory GL window, normally `10` to `15`.
- `--lambda-base`: data fidelity strength. Lower values smooth more strongly.
- `--msa-threshold`: minimum local slope anomaly threshold in slope-layer
  units.
- `--max-iter` and `--tol`: PDHG iteration cap and early-stop tolerance.
- `--convolution-method`: `direct`, `fft`, or `auto`. Short GL windows use
  direct separable convolutions by default.
- `--skip-postprocess`: disable IMF when comparing the raw FTV result.

## Delivered full-grid artifacts

The verified run for `relief_masked.nc` is stored under `artifacts\full`:

- `relief_masked_ftv.nc`: source layers plus corrected DEM, MSA mask, and SARP;
- `relief_masked_ftv_comparison.png`: final 600 DPI before/after comparison;
- `relief_masked_ftv_diagnostics.png`: correction map and slope histogram;
- `relief_masked_ftv_slope_comparison.png`: slope before/after and slope
  delta report;
- `relief_masked_ftv_3d_comparison.png`: Forge3D full-grid before/after render;
- `relief_masked_ftv_3d_roi_comparison.png`: Forge3D detailed ROI render;
- `forge3d-roi-viewer\turntable_comparison.webp`: animated ROI turntable;
- `forge3d-roi-viewer\index.html`: local interactive layer-comparison viewer;
- `snapshots\iteration_0010.png` and `snapshots\iteration_0020.png`:
  intermediate convergence comparisons;
- `relief_masked_ftv.metrics.json`: machine-readable validation metrics;
- `RUN_REPORT.md`: readable run and validation summary.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```


```

docker run -d --rm `
  --name my-forwarder `
  -p 9921:9921 `
  --add-host host.docker.internal:host-gateway `
  alpine/socat `
  tcp-listen:9921,fork,reuseaddr tcp-connect:host.docker.internal:8765
```

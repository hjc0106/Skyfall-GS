<p align="center">
  <h1 align="center"><img src="./assets/logo.png" align="top" width="38" height="38" />&nbspSkyfall-GS: Synthesizing Immersive 3D Urban Scenes from Satellite Imagery</h1>
  <p align="center">
    <a href="https://jayinnn.dev/"><strong>Jie-Ying Lee</strong></a> ·
    <a href="https://www.linkedin.com/in/yi-ruei-liu"><strong>Yi-Ruei Liu</strong></a> ·
    <a href="https://www.linkedin.com/in/shr-ruei-tsai"><strong>Shr-Ruei Tsai</strong></a> ·
    <a href="https://openreview.net/profile?id=~Wei-Cheng_Chang3"><strong>Wei-Cheng Chang</strong></a> ·
    <a href="https://kkennethwu.github.io/"><strong>Chung-Ho Wu</strong></a>
    <br>
    <a href="https://jiewenchan.github.io/"><strong>Jiewen Chan</strong></a> ·
    <a href="https://ericzzj1989.github.io/"><strong>Zhenjun Zhao</strong></a> ·
    <a href="https://hubert0527.github.io/"><strong>Chieh Hubert Lin</strong></a> ·
    <a href="https://yulunalexliu.github.io/"><strong>Yu-Lun Liu</strong></a>
  </p>
  <h3 align="center"><strong>ECCV 2026</strong> | <a href="https://skyfall-gs.jayinnn.dev/">🌐 Project Page</a> | <a href="https://arxiv.org/abs/2510.15869">📄 Paper</a> | <a href="https://huggingface.co/datasets/jayinnn/Skyfall-GS-datasets">🤗 Datasets</a> | <a href="https://huggingface.co/datasets/jayinnn/Skyfall-GS-eval">🤗 Eval Data</a> | <a href="https://huggingface.co/jayinnn/Skyfall-GS-ply">🤗 PLY Models</a></h3>
</p>
<div align="center">
  <a href="https://www.youtube.com/watch?v=zj2-aGSe6ao">
    <img src="https://img.youtube.com/vi/zj2-aGSe6ao/hqdefault.jpg" alt="Skyfall-GS Teaser Video" width="75%">
  </a>
</div>

<br>

> Synthesizing large-scale, explorable, and geometrically accurate 3D urban scenes is a challenging yet valuable task in providing immersive and embodied applications. The challenges lie in the lack of large-scale and high-quality real-world 3D scans for training generalizable generative models. In this paper, we take an alternative route to create large-scale 3D scenes by synergizing the readily available satellite imagery that supplies realistic coarse geometry and the open-domain diffusion model for creating high-quality close-up appearances. We propose **Skyfall-GS**, the first large-scale 3D-scene creation framework without costly 3D annotations, also featuring real-time, immersive 3D exploration. We tailor a curriculum-driven iterative refinement strategy to progressively enhance geometric completeness and photorealistic textures. Extensive experiments demonstrate that Skyfall-GS provides improved cross-view consistent geometry and more realistic textures compared to state-of-the-art approaches.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Running on Custom Datasets](#running-on-custom-datasets)
- [Training](#training)
  - [Stage 1: Reconstruction](#stage-1-reconstruction)
  - [Stage 2: Synthesis with Iterative Dataset Update (IDU)](#stage-2-synthesis-with-iterative-dataset-update-idu)
  - [Independent zoom generation (`dev-gen`)](#independent-zoom-generation-dev-gen)
- [Automated Training Scripts](#automated-training-scripts)
- [Pipeline Manifest](#pipeline-manifest)
- [Fused PLY for Visualization](#fused-ply-for-visualization)
- [Evaluation](#evaluation)
- [Rendering and Visualization](#rendering-and-visualization)
- [Online Viewer](#online-viewer)
- [Useful Scripts](#useful-scripts)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)

## Installation

1.  **Clone this branch of the fork (not only the upstream repository):**

    ```bash
    git clone --branch dev-stage2-gaussianzoom --recurse-submodules \
        https://github.com/hjc0106/Skyfall-GS.git
    cd Skyfall-GS
    ```

    `--recurse-submodules` initializes all submodules. Verify the MoGe
    submodule is at its pinned commit:

    ```bash
    git -C submodules/MoGe rev-parse HEAD
    # aca893a40faf6905cb722d6614cc37cb87ea06f7
    ```

    Then apply the local MoGe repair patch. The repair is shipped as a patch,
    not as an additional submodule commit; the public upstream pin is unchanged:

    ```bash
    git -C submodules/MoGe apply ../../patches/moge-sparse-depth.patch
    ```

    After applying, `git status` inside `submodules/MoGe` shows two modified
    files (`idu_depth.py`, `moge/utils/geometry_torch.py`) on top of the
    pinned commit — a **dirty worktree by applied patch** is the expected
    state. The patch recovers sparse valid pixels missed by MoGe's 64x64
    solver sampling; a genuinely empty prediction remains undefined and
    contributes no pseudo-depth loss, rather than crashing SciPy or inventing
    a depth estimate. Do not commit inside the submodule.

2.  **Create and activate a Conda environment:**

    ```bash
    conda create -y -n skyfall-gs python=3.10
    conda activate skyfall-gs
    ```

3.  **Install dependencies:**

    ```bash
    conda install cuda-toolkit=12.8 cuda-nvcc=12.8 -c nvidia

    pip install -r requirements.txt

    pip install --force-reinstall torch torchvision torchaudio

    pip install submodules/diff-gaussian-rasterization-depth
    pip install submodules/simple-knn
    pip install submodules/fused-ssim
    ```

    Environment prerequisites beyond `requirements.txt` (these are real
    configuration steps, not one-command setup):

    - **Isolated generation environments.** Stage 2 synthesis shells out to
      two pretrained models in their own interpreters. `VLM_PYTHON` must be a
      Python whose Transformers supports `Qwen3VLForConditionalGeneration`
      (verified with 4.57.6), with `VLM_MODEL_PATH` pointing at local
      Qwen3-VL-4B-Instruct weights. `DLORAL_PYTHON` points at a DLoRAL
      environment and `DLORAL_WEIGHT_ROOT` at its local weight root. All four
      are required environment variables for Stage 2.
    - **Remote training host.** The full-dataset queue runs on a remote GPU
      host reached via the SSH config aliases `remote_host` /
      `remote_read_host` in the pipeline manifest; configure them in
      `~/.ssh/config` (no credentials go into any repository file).
    - **Local paths have defaults.** Several scripts default to
      machine-specific paths (`SKYFALL_PYTHON`, `EVAL_PYTHON`, manifest and
      archive-root defaults) and all of them accept environment-variable or
      CLI overrides — see [Pipeline Manifest](#pipeline-manifest).
    - `HF_HUB_OFFLINE=1` avoids a network cache check at every MoGe model
      load once its checkpoint is cached locally. Qwen and DLoRAL already
      require local assets.

> **Note:** `submodules/MoGe` stays pinned to
> `aca893a40faf6905cb722d6614cc37cb87ea06f7`; the sparse/empty-depth repair
> lives in `patches/moge-sparse-depth.patch` and must be applied after every
> fresh clone. Other submodule pins are unchanged by this branch.

This development snapshot retains the current runtime behavior. Read the
[known boundary issues](CHANGELOG.md#known-boundary-issues-retained-in-this-development-snapshot)
before unattended archival or cleanup; use `watch --no-cleanup` when remote
deletion is not required.

## Dataset

The datasets required to train the Skyfall-GS model should be placed in the `data/` directory.

### Downloading the Datasets

The JAX and NYC datasets are available for download from Hugging Face or Google Drive.

1.  **Download the zip files:**

    [Download from Hugging Face 🤗](https://huggingface.co/datasets/jayinnn/Skyfall-GS-datasets) *(recommended)*

    [Download from Google Drive](https://drive.google.com/drive/folders/1Uugwpf7n5fj7k4UJRBuKUyrmkYcDRScQ?usp=drive_link)

2.  **Unzip the datasets into the `data/` directory:**

    ```bash
    unzip datasets_JAX.zip
    unzip datasets_NYC.zip
    ```

### Directory Structure

After unzipping, the directory structure inside the `data/` directory should look like this:

```
data/
├── datasets_JAX/
│   ├── JAX_004
│   ├── JAX_068
│   └── ...
└── datasets_NYC/
    ├── NYC_004
    ├── NYC_010
    └── ...
```

## Running on Custom Datasets

We supports training on custom datasets from two sources: COLMAP reconstructions and satellite imagery. For detailed preprocessing instructions, please refer to the [SatelliteSfM repository](https://github.com/jayin92/SatelliteSfM).

### Data Format Requirements

Your custom dataset should have the following structure to work with Skyfall-GS:

```
your_dataset/
├── images/                    # RGB images
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── masks/                   # Binary masks for valid pixels (optional: if not provided, all non-black pixels are considered valid)
│   ├── *.npy               # NumPy format (for processing)
│   ├── *.png               # PNG format (for visualization)
│   └── ...
├── transforms_train.json      # Training camera parameters
├── transforms_test.json       # Testing camera parameters (optional)
└── points3D.txt              # 3D point cloud
```

## Training

The training process is divided into two main stages.

### Stage 1: Reconstruction

This stage focuses on reconstructing the initial 3D scene from satellite imagery.

```bash
python train.py \
    -s ./data/datasets_JAX/JAX_068/ \
    -m ./outputs/JAX/JAX_068 \
    --eval \
    --port 6209 \
    --kernel_size 0.1 \
    --resolution 1 \
    --sh_degree 1 \
    --appearance_enabled \
    --lambda_depth 0 \
    --lambda_opacity 10 \
    --densify_until_iter 21000 \
    --densify_grad_threshold 0.0001 \
    --lambda_pseudo_depth 0.5 \
    --start_sample_pseudo 1000 \
    --end_sample_pseudo 21000 \
    --size_threshold 20 \
    --scaling_lr 0.001 \
    --rotation_lr 0.001 \
    --opacity_reset_interval 3000 \
    --sample_pseudo_interval 10
```

### Stage 2: Synthesis with Iterative Dataset Update (IDU)

Stage 2 starts from the Stage 1 model and continues optimizing it with the
original IDU loop. Every episode samples a set of low-elevation orbit cameras
using the existing curriculum, renders all target views, and generates refined
training images. The example below uses 10k iterations per episode, 75% generated
views plus 25% original satellite views, multiple samples per target, and MoGe
pseudo-depth supervision.

Synthesis is now GaussianZoom-inspired instead of FlowEdit/Difix3D. For each
episode:

- Novel orbit **poses** are rendered at every target view. There is no focal
  zoom: the config resolution/upscale stay 1 and `zoom_factor` stays 1.
- Source neighbors are other views of the same episode (same raster size).
  During synthesis, original training views provide the satellite context image.
  A source is chosen by depth/alpha-valid reprojection coverage within the neighbor pool,
  excluding same-pose duplicates (duplicate IDs/poses included).
- Every target view is refined; there is no single selected target. Each view
  gets geometry-guided DLoRAL refinement with its own per-view Qwen3-VL prompt,
  cached per episode/view; the `--idu_num_samples_per_view` samples of a view
  share one prompt. Descriptions summarize short, distinct feature categories
  rather than enumerating every building, to keep crowded-view JSON responses
  within the VLM output budget.
- When reprojection coverage falls below `--idu_min_reprojection_coverage`, the
  view explicitly falls back to `target_only` DLoRAL (recorded in the episode
  metadata); it is never left as the unchanged RGB render.
- Render/geometry tensors are released before pretrained generation starts, and
  intermediates (RGB/depth/alpha/flow) are stored on disk. Generated samples are
  cached keyed by checkpoint, camera, RGB, geometry, model config, and seed.
- Qwen3-VL and DLoRAL run in dedicated isolated environments (`VLM_PYTHON`,
  `DLORAL_PYTHON`, `DLORAL_WEIGHT_ROOT`) and never inside the training process.
- An episode loads Qwen once for all prompts, exits that worker, then loads one
  DLoRAL worker for all samples. Geometry and `target_only` requests share that
  model with fresh per-request flow hooks and seeds. Cache-only phases do not
  start workers, and both workers exit before MoGe/3DGS optimization.
- TensorBoard keeps the same full-resolution image pixels and metrics, but uses
  fast lossless PNG encoding and native `crc32c` from `requirements.txt`.
  Depth previews are computed only when they will be logged.

```bash
export VLM_PYTHON=$HOME/miniconda3/envs/fixanything/bin/python3.10
export VLM_MODEL_PATH=./weights/Qwen3-VL-4B-Instruct
export DLORAL_PYTHON=$HOME/miniconda3/envs/dloral/bin/python
export DLORAL_WEIGHT_ROOT=./weights/dloral

python train.py \
    -s ./data/datasets_JAX/JAX_068/ \
    -m ./outputs/JAX_idu/JAX_068 \
    --start_checkpoint ./outputs/JAX/JAX_068/chkpnt30000.pth \
    --iterative_datasets_update \
    --eval \
    --port 6209 \
    --kernel_size 0.1 \
    --resolution 1 \
    --sh_degree 1 \
    --appearance_enabled \
    --lambda_depth 0 \
    --lambda_opacity 0 \
    --opacity_reset_interval 10000000 \
    --idu_opacity_reset_interval 5000 \
    --idu_num_samples_per_view 2 \
    --densify_grad_threshold 0.0002 \
    --datasets_type jax_v1 \
    --idu_num_cams 6 \
    --idu_grid_size 3 \
    --idu_grid_width 512 \
    --idu_grid_height 512 \
    --idu_episode_iterations 10000 \
    --idu_opacity_cooling_iterations 500 \
    --lambda_pseudo_depth 0.5 \
    --idu_densify_until_iter 9000 \
    --idu_train_ratio 0.75
```

`scripts/train_stage2_gaussianzoom.sh` wraps the same invocation for a single
scene; it requires `START_CHECKPOINT`, `SOURCE_PATH`, and `OUTPUT_DIR`.

```bash
# Set the interpreter/weight variables above to the actual remote environment.
export SKYFALL_PYTHON=/path/to/skyfall-env/bin/python
export START_CHECKPOINT=/path/to/stage1/chkpnt30000.pth
export SOURCE_PATH=/path/to/data/datasets_JAX/JAX_068
export OUTPUT_DIR=/path/to/outputs/JAX_gaussianzoom_stage2/JAX_068
bash scripts/train_stage2_gaussianzoom.sh
```

The wrapper accepts extra `train.py` arguments. `IDU_RENDER_SIZE` (1024),
`IDU_EPISODE_ITERATIONS` (10000), `IDU_DENSIFY_UNTIL_ITER` (9000), and `IDU_SEED`
(0) are also configurable. Generation keeps the selected IDU image raster; it
does not enlarge the pixel grid or create a new LoD layer.

After the MoGe checkpoint is already cached locally, setting
`HF_HUB_OFFLINE=1` avoids a network cache check at every model load. Qwen and
DLoRAL already require local assets. For CPU-quota-limited containers, set
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to an appropriate
value for the quota rather than the host CPU count (the verified run used 8).

Each `idu/episode_*/` directory contains `stage2_prepared.json` (camera and
geometry inputs), `stage2_synthesis.json` (per-sample seeds, backend, cache hits,
fallbacks, prompt/refinement phase timings and actual worker metadata),
`context/prompt_*.json`, `geometry/`, `render/`, `render_refine/`, `render_depth/`,
and `render_after_train/`. Completed generated samples
are reused on a matching rerun; this is not a promise of automatic recovery of
an interrupted 3DGS optimizer step. Checkpoint, model, geometry and input-image
changes invalidate generation cache keys.

Stage 2 does not use focal-length zoom or the standalone LoD/zoom entries, which
stay independent (`train_zoom_gen.py`, `train_zoom_mvp.py`,
`scripts/run_lod_scale_chain.py`). The GaussianZoom-derived synthesis is a
GaussianZoom-inspired replacement, not a paper-complete replication, and no
quality improvement is claimed.

Remote verification on JAX_068 / RTX 4090 D (48 GB) completed the five-elevation
smoke course (85, 75, 65, 55, 45 degrees; 1024px, two views, two samples, 20
optimization steps per episode). Model, camera, loss and training-resolution
settings were not reduced for the following same-request performance comparisons:

| Measured workload | Reload-per-request / old logging | Reused workers / optimized logging |
| --- | ---: | ---: |
| Four DLoRAL requests, including target-only fallback | 72.9 s | 22.0 s |
| Three Qwen requests | 46.4 s | 26.1 s |
| Seven-view TensorBoardX evaluation | 60.2 s | 18.3 s |

All four generated images matched the original worker pixel-for-pixel, all
three Qwen outputs matched, and the 28 logged 2048px images plus evaluation
metrics were identical. These measurements verify runtime equivalence and
throughput, not final extrapolation quality or a full-training speedup guarantee.

### Independent zoom generation (`dev-gen`)

`train_zoom_gen.py` is the **old training-view zoom/LoD workflow**, kept
independent of the GaussianZoom-inspired Stage 2 above (which uses no focal
zoom and no LoD layer). It is a progressive zoom entry point that keeps
`train_zoom_mvp.py` as the baseline and delegates image generation to the
backends in `refinement/`. Use it to reproduce or compare the legacy
zoom-generation experiments, not for Stage 2 training.

```bash
python train_zoom_gen.py \
    --start_checkpoint ./outputs/JAX/JAX_068/chkpnt30000.pth \
    --output_dir ./outputs/JAX_zoom_gen/JAX_068 \
    --view_index 0 \
    --roi_center_x 0.5 --roi_center_y 0.5 \
    --roi_width 0.1 --roi_height 0.1 \
    --zoom_factors 2,4,8 \
    --sr_scale 1 \
    --supervision_mode original \
    --refine_backend unsharp
```

`zoom_factors` controls the focal-length/camera course. `sr_scale` controls
the generated image pixel multiplier and is independent of the camera zoom.
Use `--supervision_mode highres` to retain SR pixels and update the supervision
camera's image size while keeping its FoV and normalized principal point.

The comparable generation modes are:

- `flowedit`: existing FlowEdit baseline.
- `reuse`: use `--refined_image` or a prior run directory.
- `sr_fixed`: text-conditioned SR with command-line prompts.
- `sr_vlm`: text-conditioned SR with live local Qwen3-VL via
  `--vlm_model_path`, or a recorded structured result via `--prompt_json`.
- `unsharp`: deterministic offline smoke-test backend.

Geometry diagnostics are enabled by default (`--geometry_neighbor_count 2`). Each
level estimates a world-space target from zoom depth/alpha, projects it into
neighbor cameras, renders neighbor RGB/depth/alpha, and writes
`geometry.json`, spatial ROI overlays, reprojection images, validity masks, and
RGB overlays. Use `--skip_geometry` to reproduce the single-view path exactly.
Add `--multiview_refine` only when the geometry-aligned neighbor RGB should be
fused before the selected backend; this is RGB fusion, not feature-level
propagation.

Each completed level stores `wide.png`, `render_before.png`, `prompt.json`,
`refined.png`, `render_after.png`, `metrics.json`, and a level checkpoint under `zoom_<factor>x/`.

## Automated Training Scripts

The `scripts/` directory contains scripts for automated training on different datasets and configurations.

-   `scripts/run_jax.py`: Runs Stage 1 training for the JAX dataset scenes.
-   `scripts/run_jax_idu.py`: Runs Stage 2 (IDU) training for the JAX dataset scenes.
-   `scripts/run_jax_naive.py`: Runs a naive training for the JAX dataset scenes without advanced features.
-   `scripts/run_nyc.py`: Runs Stage 1 training for the NYC dataset scenes.
-   `scripts/run_nyc_idu.py`: Runs Stage 2 (IDU) training for the NYC dataset scenes.
-   `scripts/run_nyc_naive.py`: Runs a naive training for the NYC dataset scenes.
-   `scripts/train_stage2_gaussianzoom.sh`: Runs GaussianZoom-inspired Stage 2 IDU for a single scene; requires `START_CHECKPOINT`, `SOURCE_PATH`, and `OUTPUT_DIR`.

### Full-dataset training, local evaluation, and selective archival

The full workflow covers all eight JAX scenes and all four NYC scenes, not just
the four-scene JAX subset in the older launch scripts. A shared
`pipeline_manifest.json` specifies data roots, GPU ownership and artifact paths
(see [Pipeline Manifest](#pipeline-manifest) and the checked-in example
`configs/dataset_pipeline.example.json`). Qualified existing stages can be
reused instead of retrained.

- `scripts/train_full_dataset.py` runs the remote GPU queue: Stage 1 at 30k,
  followed by the original five Stage 2 episodes at 10k each. JAX uses a 3x3 IDU
  grid; NYC keeps its 4x4 grid and scene-family loss settings.
- `--compact_retention` keeps final models, metrics, prompt/camera provenance and
  representative per-episode comparisons. Consumed dense generation scratch and
  superseded Stage 2 model pairs are removed only after the newer artifacts and
  compact effects are safely retained. It is off for the original full-artifact
  entrypoints. Intermediate episodes require replaying earlier training to recreate.
- `scripts/archive_dataset_results.py --archive-root <root> watch` transfers
  selected completed stages, verifies SHA256, and publishes archive readiness.
  The archive worker is the sole remote-final cleanup owner; run the training
  queue with `--no-reap`. Cleanup requires matching archive/evaluation/model
  identities and a successful local model load. Stage 1 remains remote until its
  same-scene Stage 2 has completed.
  Already-absent paths are not counted as new deletions. Cumulative removal totals
  come from the deduplicated ledger and describe logical file bytes, not physical
  disk space freed (hard-linked files may still have another retained link).
- `scripts/run_eval_queue.sh --exit-when-complete` consumes verified local archives
  on the local GPU while the remote GPU trains the next stage. Checkpoints load
  through CPU so optimizer state does not occupy inference VRAM.

Keep **both the final checkpoint and its matching PLY**: learned appearance state
is in the checkpoint, while `filter_3D` is in the PLY. Raw datasets and pretrained
generation weights are not duplicated in each result archive.

Evaluation retains per-view and aggregate PSNR/SSIM/LPIPS on the held-out satellite
cameras, fixed-trajectory videos and Stage1/Stage2 comparison panels. Official
external GT is currently available for four JAX and four NYC scenes; missing GT
for the remaining JAX scenes is explicit. Distributional CLIP-FID/CMMD,
similarity to released baseline renders, and appearance-inferred GT pairings are
separate scopes. Camera-unverified GT pairing is disabled by default and is not
reported as a verified reconstruction metric.

After diagnosing and fixing a failed stage, explicitly retry only its scene:

```bash
python scripts/train_full_dataset.py run --only JAX_164 \
    --retry-failed --wait-lock --max-attempts 2 --no-reap
```

Use the same control/run roots as the active queue. `--wait-lock` waits for the
entire active controller to exit; it neither interrupts its trainer nor rewrites
live queue state. Once it owns the lock, the recovery controller requeues selected
failed stages below the attempt ceiling and then runs their pending dependent stages.
Completed/cleared stages and unselected failures are not reset. Without
`--wait-lock`, a competing controller is refused immediately.

## Pipeline Manifest

The full-dataset workers share one deployment configuration file. A
structural example with all contract values and user-editable placeholders is
checked in at [`configs/dataset_pipeline.example.json`](configs/dataset_pipeline.example.json).
Copy it to your deployment archive root and edit the marked paths:

```bash
mkdir -p /path/to/experiments/stage2_gaussianzoom
cp configs/dataset_pipeline.example.json \
   /path/to/experiments/stage2_gaussianzoom/pipeline_manifest.json
```

Fields to customize (documented in the `_documentation` block inside the
example): `local_project`, `local_python`, `archive_root`,
`local_dataset_root`, `local_eval_archive_root`, `remote_host`,
`remote_read_host`, `remote_project`, `remote_run_root`,
`remote_control_root`, `remote_activation`. `remote_host` /
`remote_read_host` are SSH **config aliases** from `~/.ssh/config`; no
credentials belong in the manifest or any repository file. Keep the 12-scene
list, the training recipe (`stage1_iterations` 30000, five 10000-iteration
Stage 2 episodes, `final_iteration` 80000) and the retention / publish
contracts unchanged unless you know what a worker reads.

**Copying the example is not turnkey.** `scripts/train_full_dataset.py` is a
frozen experiment queue, not a manifest-driven generic launcher:

- Its authoritative scene table (`SCENES`) and per-recipe argument lists are
  **embedded in the script**; `plan --manifest` cross-checks the shared
  contract (scene list/order, training block, stage-publish contract). The
  `run` subcommand does not accept `--manifest` or derive its job arguments from it.
- Runtime roots are frozen to the 2026-09-13 `stage2_gaussianzoom`
  experiment. A fresh deployment MUST pass its own
  `--run-root`/`--control-root`/`--archive-root` (or env `GZ_RUN_ROOT`,
  `GZ_CONTROL_ROOT`, `GZ_ARCHIVE_ROOT`). Use `plan --manifest` to cross-check
  the shared configuration before launching `run`.
- Reuse is a snapshot configuration, not derived from the manifest: the
  embedded `reuse_stage1`/`reuse_stage2` entries for JAX_068 point at frozen
  **remote** absolute paths of the qualified runs
  (`jax068_gszoom_20260912_01/stage1`,
  `jax068_stage2_gz_20260913_082647/full`), so `run` hard-links those verified
  finals instead of retraining JAX_068. A new deployment with different
  snapshot paths must reconfigure the embedded scene table (or verify its own
  reused run first); `publish-reuse` only publishes markers for these
  embedded reuse jobs.

Which tool reads which configuration, and where results land:

| Worker | Configuration input | Output |
| --- | --- | --- |
| `scripts/train_full_dataset.py` (remote GPU) | frozen experiment queue: scene table and recipes are **embedded in the script**; `plan --manifest` cross-checks; `run` takes root overrides via `--run-root`/`--control-root`/`--archive-root` (env `GZ_RUN_ROOT`, `GZ_CONTROL_ROOT`, `GZ_ARCHIVE_ROOT`); JAX_068 reuse paths are frozen embedded snapshot paths | `<remote_run_root>/<scene>/<stage>/stage_complete.json` marker |
| `scripts/archive_dataset_results.py` (remote + local) | **requires** `<archive-root>/pipeline_manifest.json`; `--archive-root` (env `SKYFALL_ARCHIVE_ROOT`); `--host` defaults to manifest `remote_host` | `<archive_root>/<scene>/<stage>/archive_status.json` (`status: verified`) after SHA256 verification on both ends |
| `scripts/evaluate_dataset.py` / `scripts/run_eval_queue.sh` (local GPU) | `--manifest` / env `EVAL_MANIFEST` (also reads `local_dataset_root`) | `<archive_root>/<scene>/evaluation/<stage>/evaluation_status.json` (`model_load_verified: true`, then `completed`) |
| `scripts/local_scene_comparison.py` + `scripts/compose_scene_comparison_video.py` | verified local archives under the archive root | catalog + one labelled comparison video (see [Single local comparison video](#single-local-comparison-video)) |

Required environment/tool/model assets before starting: the four Stage 2
generation variables (`VLM_PYTHON`, `VLM_MODEL_PATH`, `DLORAL_PYTHON`,
`DLORAL_WEIGHT_ROOT`), the JAX/NYC datasets under `local_dataset_root` (and
on the remote host under the training project), official eval data for
`local_eval_archive_root`, and SSH alias connectivity to the remote host.
Never commit datasets, model weights, videos or credentials to Git.

### Ground-truth and protocol limitations

- Official external GT exists for JAX_004/068/214/260 and NYC_004/010/219/336;
  the remaining JAX scenes have no official GT and are reported as missing,
  never silently dropped.
- The formal heldout metrics protocol is the default fixed appearance row
  `min(6, n_train - 1)`; mean-embedding appearance is a different protocol and
  the two must not be compared as one number.
- Original released Skyfall-GS Stage2 references are the upstream authors'
  models. Comparisons against them are external; JAX_168/175 comparisons are
  internal Stage1/Stage2 only. The final six-scene video excludes JAX_264.

## Fused PLY for Visualization

Do not directly use the raw `.ply` files under the training output directory for online visualization.
After training, you should fuse the model first, then visualize the fused file.

Pre-built fused PLY files for all scenes are also available for direct download from Hugging Face:

[Download fused PLY from Hugging Face 🤗](https://huggingface.co/jayinnn/Skyfall-GS-ply)

1.  **Generate a fused PLY file from a trained model:**

    ```bash
    python create_fused_ply.py \
        -m ./outputs/JAX_idu/JAX_068 \
        --output_ply ./fused/JAX_068_fused.ply \
        --iteration 80000 \
        --load_from_checkpoints
    ```

2.  **Use the fused PLY (`*_fused.ply`) for visualization/rendering tools that expect a standalone PLY.**

## Evaluation

The `eval.py` script is used for evaluating the performance of a trained model. It computes various metrics by comparing the rendered images with ground truth images.

### Downloading Evaluation Data

The evaluation data, which includes the ground truth videos and the rendered videos from other methods, can be downloaded from Hugging Face or Google Drive.

[Download from Hugging Face 🤗](https://huggingface.co/datasets/jayinnn/Skyfall-GS-eval) *(recommended)*

[Download from Google Drive](https://drive.google.com/drive/folders/1hSFe9yGOwJCLBK7ZLHB-49_x73Ebk_VV?usp=drive_link)

After downloading, unzip the file and place the `results_eval` directory in the root of the project.

### Usage

```bash
python eval.py \
    --data_dir results_eval/data_eval_JAX \
    --temp_dir temp_frames_JAX \
    --methods mip-splatting sat-nerf eogs corgs ours_stage1 ours_stage2 \
    --output_file metrics_results_JAX.csv \
    --frame_rate 30 \
    --resolution 1024 \
    --batch_size 64 

python eval.py \
    --data_dir results_eval/data_eval_NYC \
    --temp_dir temp_frames_NYC \
    --methods citydreamer gaussiancity corgs ours_stage1 ours_stage2 \
    --output_file metrics_results_NYC.csv \
    --frame_rate 24 \
    --no_resize \
    --batch_size 64
```

The script calculates the following metrics:
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **CLIP-FID**: FID score calculated using CLIP features
- **CMMD**: CLIP Maximum Mean Discrepancy

## Rendering and Visualization

The `render_video.py` script can be used to render a video from a trained model using a specified camera path.

```bash
python render_video.py \
    -m <path_to_model_directory> \
    --camera_path <path_to_camera.json> \
    --load_from_checkpoints \
    --iteration <checkpoint_iteration> \
    --save_images \
    --depth
```

-   `-m`: Path to the model directory.
-   `--camera_path`: Path to the camera trajectory JSON file.
-   `--load_from_checkpoints`: Load the model from a checkpoint.
-   `--iteration`: The checkpoint iteration to use.
-   `--save_images`: Save individual frames of the video.
-   `--depth`: Render depth maps instead of RGB images.

You can also render a video from a `.ply` file using `render_video_from_ply.py`:
```bash
python render_video_from_ply.py \
    --ply_path <path_to_ply_file> \
    --camera_path <path_to_camera.json>
```

For models trained with Skyfall-GS, use a fused PLY generated by `create_fused_ply.py` (not the raw training-output `.ply` files).

### Single local comparison video

From the local archive snapshot, render fresh CUDA orbits and compose one
labelled video (no web server). Existing orbit assets can be reused by running
only the second command.

```bash
OUT=/path/to/local_scene_comparison
python scripts/local_scene_comparison.py build --output-root "$OUT"
python scripts/compose_scene_comparison_video.py \
    --catalog "$OUT/catalog.json" \
    --output "$OUT/skyfall_vs_gaussianzoom_6scenes.mp4"
```

The composer selects six scenes with both complete stages and produces an
84-second, 1920x1080 H.264 video with scene chapters and no closing explanation.
It distinguishes this experiment's Stage1/Stage2 comparison from comparison
against the original Skyfall-GS released Stage2 videos. Original references are
available locally for JAX_004/068/214/260; JAX_168/175 use internal stage
comparisons. JAX_264's Stage1-only segment is excluded. Camera/model provenance
and supplementary held-out metrics are stored beside the video; formal archive
evaluation records and model files are not overwritten. Matching a named camera
trajectory is not a claim of verified pixel-aligned GES ground truth.

## Online Viewer

Use the fused PLY generated in [Fused PLY for Visualization](#fused-ply-for-visualization) for online viewing.

### Option 1: Mip-Splatting Viewer

Use the [online viewer](https://niujinshuchong.github.io/mip-splatting-demo).

For optimal viewing, use the following settings:
-   **Up vector:** `0,0,1`
-   **SH degree:** `1`
-   **Camera origin:** `0,0,200`

### Option 2: SuperSplat (Alternative)

You can also use [SuperSplat Editor](https://superspl.at/editor), an open-source web-based Gaussian splat editor/viewer from PlayCanvas.

Recommended workflow:
1.  Generate `*_fused.ply` first (see [Fused PLY for Visualization](#fused-ply-for-visualization)).
2.  Open `https://superspl.at/editor`.
3.  Import the fused PLY by drag-and-drop or via `File` > `Import`.
4.  (Optional) Publish with `File` > `Publish`, or export a standalone viewer app from `File` > `Export`.

References:
-   SuperSplat repo: https://github.com/playcanvas/supersplat
-   Import/Export docs: https://developer.playcanvas.com/user-manual/gaussian-splatting/editing/supersplat/import-export/

## Useful Scripts

This project includes several other useful scripts:

-   `align_ges.py`: Find optimal target altitude by comparing with ground truth.
-   `convert.py`: A COLMAP converter script.
-   `dsmr.py`: Functions for DSM registration.
-   `evaluate_gs_geometry.py`: Evaluate geometry accuracy for a single scene.
-   `gen_render_path.py`: Generate a camera path for an orbit view around a target point.
-   `render_videos.py`: A script for batch rendering of videos from multiple models and camera paths.
-   `sat_utils.py`: Utility functions for handling satellite images and georeferenced data.
-   `scripts/merge_images.py`: Merge two frames into one.

## Acknowledgement

This codebase is built upon the following open-source projects:
-   [Mip-Splatting](https://github.com/autonomousvision/mip-splatting)
-   [WildGuassians](https://github.com/jkulhanek/wild-gaussians)
-   [FlowEdit](https://github.com/fallenshock/FlowEdit)
-   [MoGe](https://github.com/microsoft/MoGe)
-   [SatelliteSfM](https://github.com/Kai-46/SatelliteSfM)

We thank the authors for their contributions.

This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2 and 113-2628-EA49-023-. The authors are grateful to Google, NVIDIA, and MediaTek Inc. for their generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{lee2025SkyfallGS,
  title = {{Skyfall-GS}: Synthesizing Immersive {3D} Urban Scenes from Satellite Imagery},
  author = {Jie-Ying Lee and Yi-Ruei Liu and Shr-Ruei Tsai and Wei-Cheng Chang and Chung-Ho Wu and Jiewen Chan and Zhenjun Zhao and Chieh Hubert Lin and Yu-Lun Liu},
  journal = {arXiv preprint},
  year = {2025},
  eprint = {2510.15869},
  archivePrefix = {arXiv}
}
```

## License

This project is licensed under the terms of the [Apache 2 License](LICENSE).

### Local Qwen3-VL prompts for zoom generation

`train_zoom_gen.py` can generate structured prompts from the wide and zoom
renderings using local Qwen3-VL-4B-Instruct weights. Add these arguments to your
existing zoom-generation command (including its checkpoint, ROI and SR settings):

```bash
--refine_backend sr_vlm \
--vlm_model_path /datacc05/hongjiacheng/qwen_hub/Qwen3-VL-4B-Instruct \
--vlm_python /home/hongjiacheng/miniconda3/envs/fixanything/bin/python \
--vlm_device cuda:0 \
--vlm_max_image_size 1024 \
--vlm_max_new_tokens 768
```

The VLM runs in a subprocess and exits before refinement, releasing its GPU
allocation. This allows the existing `skyfall-gs` training environment to retain
Transformers 4.46.3; the selected VLM interpreter must support
`Qwen3VLForConditionalGeneration` (locally verified with Transformers 4.57.6).
Weights and processor files are loaded locally only. `--vlm_device` uses the
process-visible CUDA index, respecting `CUDA_VISIBLE_DEVICES`.

Use either `--vlm_model_path` or recorded `--prompt_json`. `sr_fixed` rejects
both. Prompts include the raw model response and model/inference provenance;
shared-region semantics are supplied to later levels. Cached prompts avoid
loading the VLM again. Invalid or truncated JSON raises an error rather than
silently switching to fixed prompts; increase the token limit if necessary.
The supplied checkpoint is the standard Qwen3-VL Instruct model, not the
Chain-of-Zoom fine-tuned model used in GaussianZoom. SR still requires its own
backend/model configuration.

## Physical Resolution Field (PRF)

独立物理分辨率后处理模块已迁入 [`thirdparty/PRF`](thirdparty/PRF/README.md)。
输入米制高斯 PLY 和原始训练相机，输出观测、高斯核、采样间距及融合分辨率，
支持 GS 深度遮挡、冻结几何对照和离线 3D 可视化。安装、运行和测试见模块文档。

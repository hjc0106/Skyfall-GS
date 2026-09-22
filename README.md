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
  - [Independent zoom generation (`dev-gen`)](#independent-zoom-generation-dev-gen)
  - [Stage 1: Reconstruction](#stage-1-reconstruction)
  - [Stage 2: Synthesis with Iterative Dataset Update (IDU)](#stage-2-synthesis-with-iterative-dataset-update-idu)
- [Automated Training Scripts](#automated-training-scripts)
- [Fused PLY for Visualization](#fused-ply-for-visualization)
- [Evaluation](#evaluation)
- [Rendering and Visualization](#rendering-and-visualization)
- [Online Viewer](#online-viewer)
- [Useful Scripts](#useful-scripts)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone --recurse-submodules https://github.com/jayin92/Skyfall-GS.git
    cd Skyfall-GS
    ```

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

This stage refines the geometry and synthesizes high-quality textures using an iterative dataset update strategy. This stage uses a pretrained model from Stage 1.

```bash
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
    --idu_opacity_reset_interval 5000 \
    --idu_refine \
    --idu_num_samples_per_view 2 \
    --densify_grad_threshold 0.0002 \
    --idu_num_cams 6 \
    --idu_use_flow_edit \
    --idu_render_size 1024 \
    --idu_flow_edit_n_min 4 \
    --idu_flow_edit_n_max 10 \
    --idu_grid_size 3 \
    --idu_grid_width 512 \
    --idu_grid_height 512 \
    --idu_episode_iterations 10000 \
    --idu_iter_full_train 0 \
    --idu_opacity_cooling_iterations 500 \
    --lambda_pseudo_depth 0.5 \
    --idu_densify_until_iter 9000 \
    --idu_train_ratio 0.75
```

### Independent zoom generation (`dev-gen`)

`train_zoom_gen.py` is an independent progressive zoom entry point. It keeps
`train_zoom_mvp.py` as the baseline and delegates image generation to the
backends in `refinement/`.

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
-   [`docs/lod_jax068_five_episode.md`](docs/lod_jax068_five_episode.md): Reproduce the independent JAX_068 L0-start five-episode LoD experiment.

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

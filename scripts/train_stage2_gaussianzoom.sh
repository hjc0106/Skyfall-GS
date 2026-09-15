#!/usr/bin/env bash
# Skyfall Stage 2 (GaussianZoom): synthesize novel low-elevation orbit views with
# geometry-guided DLoRAL + per-view Qwen3-VL prompts, then run the original IDU
# episodes (10k iterations, shared low-elevation camera curriculum) on the
# refined views. This is a GaussianZoom-inspired replacement for the old
# FlowEdit/Difix3D/DreamScene IDU synthesis; it is not a paper-complete
# GaussianZoom reproduction and makes no quality-improvement claim.
#
# Required:
#   START_CHECKPOINT  Stage 1 checkpoint (e.g. .../chkpnt30000.pth)
#   SOURCE_PATH       dataset directory (transforms_train.json + images/)
#   OUTPUT_DIR        Stage 2 output directory
#
# Optional (defaults preserve the original IDU recipe):
#   STAGE2_GPU (0), STAGE2_PORT (6209), DATASETS_TYPE (jax_v1),
#   IDU_NUM_CAMS (6), IDU_GRID_SIZE (3), IDU_GRID_WIDTH (512),
#   IDU_GRID_HEIGHT (512), IDU_NUM_SAMPLES_PER_VIEW (2),
#   IDU_TRAIN_RATIO (0.75), IDU_EPISODE_ITERATIONS (10000),
#   LAMBDA_PSEUDO_DEPTH (0.5).
#
# Isolated environments (also read by arguments/__init__.py):
#   VLM_PYTHON, VLM_MODEL_PATH, DLORAL_PYTHON, DLORAL_WEIGHT_ROOT.
# GPU smoke verified on JAX_068; full-training quality remains experiment-specific.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKYFALL_PY="${SKYFALL_PYTHON:-$HOME/miniconda3/envs/skyfall-gs/bin/python}"
GPU="${STAGE2_GPU:-0}"
PORT="${STAGE2_PORT:-6209}"

START_CHECKPOINT="${START_CHECKPOINT:?set START_CHECKPOINT to the Stage 1 chkpnt30000.pth}"
SOURCE_PATH="${SOURCE_PATH:?set SOURCE_PATH to the dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to the Stage 2 output directory}"

export VLM_PYTHON="${VLM_PYTHON:-$HOME/miniconda3/envs/fixanything/bin/python3.10}"
export VLM_MODEL_PATH="${VLM_MODEL_PATH:-$ROOT/weights/Qwen3-VL-4B-Instruct}"
export DLORAL_PYTHON="${DLORAL_PYTHON:-$HOME/miniconda3/envs/dloral/bin/python}"
export DLORAL_WEIGHT_ROOT="${DLORAL_WEIGHT_ROOT:-$ROOT/weights/dloral}"

DATASETS_TYPE="${DATASETS_TYPE:-jax_v1}"
IDU_NUM_CAMS="${IDU_NUM_CAMS:-6}"
IDU_GRID_SIZE="${IDU_GRID_SIZE:-3}"
IDU_GRID_WIDTH="${IDU_GRID_WIDTH:-512}"
IDU_GRID_HEIGHT="${IDU_GRID_HEIGHT:-512}"
IDU_NUM_SAMPLES_PER_VIEW="${IDU_NUM_SAMPLES_PER_VIEW:-2}"
IDU_TRAIN_RATIO="${IDU_TRAIN_RATIO:-0.75}"
IDU_EPISODE_ITERATIONS="${IDU_EPISODE_ITERATIONS:-10000}"
LAMBDA_PSEUDO_DEPTH="${LAMBDA_PSEUDO_DEPTH:-0.5}"
IDU_DENSIFY_UNTIL_ITER="${IDU_DENSIFY_UNTIL_ITER:-9000}"
IDU_RENDER_SIZE="${IDU_RENDER_SIZE:-1024}"
IDU_SEED="${IDU_SEED:-0}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUTPUT_DIR"

exec "$SKYFALL_PY" train.py \
  -s "$SOURCE_PATH" \
  -m "$OUTPUT_DIR" \
  --start_checkpoint "$START_CHECKPOINT" \
  --iterative_datasets_update \
  --eval \
  --port "$PORT" \
  --kernel_size 0.1 \
  --resolution 1 \
  --sh_degree 1 \
  --appearance_enabled \
  --lambda_depth 0.0 \
  --lambda_opacity 0.0 \
  --opacity_reset_interval 10000000 \
  --idu_opacity_reset_interval 5000 \
  --idu_num_samples_per_view "$IDU_NUM_SAMPLES_PER_VIEW" \
  --densify_grad_threshold 0.0002 \
  --datasets_type "$DATASETS_TYPE" \
  --idu_num_cams "$IDU_NUM_CAMS" \
  --idu_render_size "$IDU_RENDER_SIZE" \
  --idu_seed "$IDU_SEED" \
  --idu_grid_size "$IDU_GRID_SIZE" \
  --idu_grid_width "$IDU_GRID_WIDTH" \
  --idu_grid_height "$IDU_GRID_HEIGHT" \
  --idu_episode_iterations "$IDU_EPISODE_ITERATIONS" \
  --idu_opacity_cooling_iterations 500 \
  --lambda_pseudo_depth "$LAMBDA_PSEUDO_DEPTH" \
  --idu_densify_until_iter "$IDU_DENSIFY_UNTIL_ITER" \
  --idu_train_ratio "$IDU_TRAIN_RATIO" \
  "$@"

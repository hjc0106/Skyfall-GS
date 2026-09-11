#!/usr/bin/env bash
# JAX_214 Stage1 L0, same flags as JAX_068 / README. 30k iterations.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKYFALL_PY="${SKYFALL_PYTHON:-$HOME/miniconda3/envs/skyfall-gs/bin/python}"
GPU="${ABSORB_GPU:-2}"
SOURCE="${STAGE1_SOURCE:-$ROOT/data/datasets_JAX/JAX_214}"
MODEL="${STAGE1_MODEL:-$ROOT/skyfall-gs_exp/stage1/JAX_214}"
PORT="${STAGE1_PORT:-6214}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$MODEL"

exec "$SKYFALL_PY" train.py \
  -s "$SOURCE" \
  -m "$MODEL" \
  --eval \
  --port "$PORT" \
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

#!/usr/bin/env bash
# Smoke RaDe-GS through imported L0. Does not train a detail layer.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="${START_CHECKPOINT:-$ROOT/skyfall-gs_exp/stage1/JAX_068/chkpnt30000.pth}"
OUT="${RADE_OUTPUT_DIR:-$ROOT/skyfall-gs_exp/lod_rade_smoke}"
GZ_ROOT="${GAUSSIANZOOM_ROOT:-$ROOT/vendor/gaussianzoom_distill}"
SKYFALL_PY="${SKYFALL_PYTHON:-$HOME/miniconda3/envs/skyfall-gs/bin/python}"
GPU="${ABSORB_GPU:-2}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
"$SKYFALL_PY" scripts/verify_lod_rade.py \
  --start_checkpoint "$CHECKPOINT" \
  --output_dir "$OUT" \
  --gz_root "$GZ_ROOT" \
  --view_index 0 \
  --roi_center_x 0.592 \
  --roi_center_y 0.53 \
  --roi_width 0.1 \
  --roi_height 0.1 \
  --zoom_factor 2 \
  --step_scale 2

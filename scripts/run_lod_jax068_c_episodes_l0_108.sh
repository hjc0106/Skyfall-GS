#!/usr/bin/env bash
# JAX_068 L0-30K start, freeze L0, five-episode shared L1, 108 views / 5K steps.
# Independent archive. Do not write into lod_jax068_c_episodes_l1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${C_EPISODES_L0_108_DIR:-$ROOT/skyfall-gs_exp/lod_jax068_c_episodes_l0_start_108}"
CHECKPOINT="${START_CHECKPOINT:-$ROOT/skyfall-gs_exp/stage1/JAX_068/chkpnt30000.pth}"
GZ_ROOT="${GAUSSIANZOOM_ROOT:-$ROOT/vendor/gaussianzoom_distill}"
SKYFALL_PY="${SKYFALL_PYTHON:-python}"
DLORAL_PY="${DLORAL_PYTHON:-python}"
VLM_PY="${VLM_PYTHON:-python}"
GPU="${ABSORB_GPU:-8}"
PHASE="${1:-probe}"
JAX068_DATASET_DIR="${JAX068_DATASET_DIR:-$ROOT/data/datasets_JAX/JAX_068}"

if [[ "$PHASE" =~ ^(probe|supervise|flowedit|episode|all)$ ]]; then
  : "${DLORAL_WEIGHT_ROOT:?Set DLORAL_WEIGHT_ROOT to the DLoRAL model directory}"
  : "${VLM_MODEL_PATH:?Set VLM_MODEL_PATH to the Qwen3-VL model directory}"
  : "${FLOWEDIT_MODEL_PATH:?Set FLOWEDIT_MODEL_PATH to the FLUX model directory}"
fi
DLORAL_WEIGHT_ROOT="${DLORAL_WEIGHT_ROOT:-}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-}"
FLOWEDIT_MODEL_PATH="${FLOWEDIT_MODEL_PATH:-}"
WEIGHT_ROOT="$DLORAL_WEIGHT_ROOT"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DLORAL_WEIGHT_ROOT="$WEIGHT_ROOT"
export VLM_MODEL_PATH
export FLOWEDIT_MODEL_PATH
export JAX068_DATASET_DIR

mkdir -p "$OUT"
echo "JAX_068 L0-start 108-view five-episode output: $OUT (GPU $GPU, phase $PHASE)"

exec "$SKYFALL_PY" scripts/run_lod_jax068_c_episodes_l0_108.py \
  --start_checkpoint "$CHECKPOINT" \
  --output_dir "$OUT" \
  --gz_root "$GZ_ROOT" \
  --phase "$PHASE" \
  --step_scale 2.0 \
  --generation_seed 4001 \
  --training_seed 0 \
  --vlm_model_path "$VLM_MODEL_PATH" \
  --vlm_python "$VLM_PY" \
  --flowedit_model_path "$FLOWEDIT_MODEL_PATH" \
  --dloral_python "$DLORAL_PY" \
  --dloral_device cuda:0 \
  --dloral_root "$ROOT/submodules/DLoRAL" \
  --sd_path "$WEIGHT_ROOT/stable-diffusion-2-1-base" \
  --ckpt "$WEIGHT_ROOT/model_enhanced.pkl" \
  --spynet "$WEIGHT_ROOT/spynet_20210409-c6c1bd09.pth" \
  "${@:2}"

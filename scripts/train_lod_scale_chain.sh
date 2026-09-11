#!/usr/bin/env bash
# Thin wrapper around the recoverable Python 2x->4x entry. Does not run 8x.
# SpyNet control: pass --spynet_control to the Python entry, not this script.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKYFALL_PY="${SKYFALL_PYTHON:-$HOME/miniconda3/envs/skyfall-gs/bin/python}"
GPU="${ABSORB_GPU:-2}"
OUT="${OUTPUT_DIR:?set OUTPUT_DIR}"
ROI_CX="${ROI_CENTER_X:?set ROI_CENTER_X}"
ROI_CY="${ROI_CENTER_Y:?set ROI_CENTER_Y}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"

exec "$SKYFALL_PY" scripts/run_lod_scale_chain.py \
  --output_dir "$OUT" \
  --start_checkpoint "${START_CHECKPOINT:-$ROOT/skyfall-gs_exp/stage1/JAX_068/chkpnt30000.pth}" \
  --roi_center_x "$ROI_CX" \
  --roi_center_y "$ROI_CY" \
  --roi_width "${ROI_WIDTH:-0.1}" \
  --roi_height "${ROI_HEIGHT:-0.1}" \
  --view_index "${VIEW_INDEX:-0}" \
  --stages "${STAGES:-all}" \
  --alignment "${ALIGNMENT:-geometry}" \
  --steps "${STEPS_PER_LEVEL:-500}" \
  --seed "${SEED:-0}" \
  --mix_ratio "${MIX_RATIO:-0.2}" \
  --gz_root "${GAUSSIANZOOM_ROOT:-$ROOT/vendor/gaussianzoom_distill}" \
  --dloral_python "${DLORAL_PYTHON:-$HOME/miniconda3/envs/dloral/bin/python}" \
  ${PROMPT_JSON:+--prompt_json "$PROMPT_JSON"} \
  ${FORCE:+--force} \
  ${ALLOW_UNLINEAGED:+--allow_unlineaged_reuse} \
  "$@"

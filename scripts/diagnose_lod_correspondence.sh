#!/usr/bin/env bash
# Depth-driven co-visible correspondence on saved 2x L1 bundles. Does not train.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${L1_OUTPUT_DIR:-$ROOT/skyfall-gs_exp/lod_l1_absorb_2x}"
CHECKPOINT="${START_CHECKPOINT:-$ROOT/skyfall-gs_exp/stage1/JAX_068/chkpnt30000.pth}"
GZ_ROOT="${GAUSSIANZOOM_ROOT:-$ROOT/vendor/gaussianzoom_distill}"
SKYFALL_PY="${SKYFALL_PYTHON:-$HOME/miniconda3/envs/skyfall-gs/bin/python}"
GPU="${ABSORB_GPU:-2}"
MODES="${ABSORB_MODES:-target_only,spynet,geometry}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"

"$SKYFALL_PY" scripts/diagnose_lod_correspondence.py \
  --start_checkpoint "$CHECKPOINT" \
  --root "$OUT_ROOT" \
  --modes "$MODES" \
  --gz_root "$GZ_ROOT" \
  --view_index 0 \
  --roi_center_x 0.592 --roi_center_y 0.53 --roi_width 0.1 --roi_height 0.1 \
  --zoom_factor 2 --step_scale 2

"$SKYFALL_PY" scripts/summarize_lod_correspondence.py --root "$OUT_ROOT"
echo "Correspondence diagnosis complete. Summary in $OUT_ROOT/lod_l1_correspondence_summary.json"

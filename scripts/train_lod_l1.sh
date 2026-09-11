#!/usr/bin/env bash
# L1 2x absorption on the joint RaDe-GS path. Same seed / mix / densify for every mode.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ALIGN="${DLORAL_ALIGN_DIR:-$ROOT/skyfall-gs_exp/zoom_gen_dloral_align_2048}"
OUT_ROOT="${L1_OUTPUT_DIR:-$ROOT/skyfall-gs_exp/lod_l1_absorb_2x}"
CHECKPOINT="${START_CHECKPOINT:-$ROOT/skyfall-gs_exp/stage1/JAX_068/chkpnt30000.pth}"
GZ_ROOT="${GAUSSIANZOOM_ROOT:-$ROOT/vendor/gaussianzoom_distill}"
SKYFALL_PY="${SKYFALL_PYTHON:-$HOME/miniconda3/envs/skyfall-gs/bin/python}"
GPU="${ABSORB_GPU:-2}"
STEPS="${STEPS_PER_LEVEL:-500}"
EVAL_STEPS="${EVAL_STEPS:-0,50,100,250,500}"
MIX_RATIO="${MIX_RATIO:-0.2}"
SEED="${SEED:-0}"
MODES="${ABSORB_MODES:-target_only,spynet,geometry}"

IFS=',' read -r -a MODE_LIST <<< "$MODES"
missing=0
for mode in "${MODE_LIST[@]}"; do
  if [[ ! -f "$ALIGN/$mode/refined.png" ]]; then
    echo "missing: $ALIGN/$mode/refined.png" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"

for mode in "${MODE_LIST[@]}"; do
  out="$OUT_ROOT/$mode"
  curve="$out/absorption_curve.json"
  if [[ -f "$curve" && "${FORCE:-0}" != "1" ]]; then
    echo "skip completed $mode ($curve exists; FORCE=1 to rerun)"
    if [[ ! -f "$out/cross_view.json" ]]; then
      echo "diagnosing $mode"
      "$SKYFALL_PY" scripts/diagnose_lod_l1.py \
        --start_checkpoint "$CHECKPOINT" \
        --lod_checkpoint "$out/l1_final.lod.pt" \
        --output_dir "$out" \
        --gz_root "$GZ_ROOT" \
        --view_index 0 \
        --roi_center_x "${ROI_CENTER_X:-0.592}" --roi_center_y "${ROI_CENTER_Y:-0.53}" \
        --roi_width "${ROI_WIDTH:-0.1}" --roi_height "${ROI_HEIGHT:-0.1}" \
        --zoom_factor 2 --step_scale 2
    fi
    continue
  fi
  if [[ "${FORCE:-0}" == "1" ]]; then
    rm -rf "$out"
  elif [[ -d "$out" && -n "$(ls -A "$out" 2>/dev/null)" ]]; then
    echo "partial output exists in $out; pass FORCE=1 to replace." >&2
    exit 1
  fi
  mkdir -p "$out"
  echo "=== L1 absorb $mode: ${STEPS} steps, mix=${MIX_RATIO}, seed=${SEED} ==="
  "$SKYFALL_PY" scripts/train_lod_l1.py \
    --start_checkpoint "$CHECKPOINT" \
    --output_dir "$out" \
    --refined_image "$ALIGN/$mode/refined.png" \
    --gz_root "$GZ_ROOT" \
    --view_index 0 \
    --roi_center_x "${ROI_CENTER_X:-0.592}" --roi_center_y "${ROI_CENTER_Y:-0.53}" \
    --roi_width "${ROI_WIDTH:-0.1}" --roi_height "${ROI_HEIGHT:-0.1}" \
    --zoom_factor 2 --step_scale 2 \
    --steps "$STEPS" --eval_steps "$EVAL_STEPS" \
    --mix_ratio "$MIX_RATIO" --seed "$SEED"
done

"$SKYFALL_PY" scripts/summarize_lod_l1.py --root "$OUT_ROOT"
echo "L1 absorption complete. Summary in $OUT_ROOT/lod_l1_summary.json"

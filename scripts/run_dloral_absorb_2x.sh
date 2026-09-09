#!/usr/bin/env bash
# Absorb the accepted 2048 tiled DLoRAL images into frozen Stage1 3DGS.
# Reuses target_only / SpyNet / geometry refined.png. Does not regenerate SR.
# Same checkpoint, ROI, 2x camera, seed, mix_ratio, and view sequence.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ALIGN="${DLORAL_ALIGN_DIR:-$ROOT/skyfall-gs_exp/zoom_gen_dloral_align_2048}"
OUT_ROOT="${ABSORB_OUTPUT_DIR:-$ROOT/skyfall-gs_exp/zoom_gen_dloral_absorb_2x}"
PROMPT_JSON="${DLORAL_PROMPT_JSON:-$ROOT/skyfall-gs_exp/zoom_gen_20260908/skyfall_zoom_gen_sr_vlm_geometry_20260908/zoom_2x/prompt.json}"
CHECKPOINT="${START_CHECKPOINT:-$ROOT/skyfall-gs_exp/stage1/JAX_068/chkpnt30000.pth}"
SKYFALL_PY="${SKYFALL_PYTHON:-$HOME/miniconda3/envs/skyfall-gs/bin/python}"
GPU="${ABSORB_GPU:-2}"
STEPS="${STEPS_PER_LEVEL:-500}"
EVAL_STEPS="${EVAL_STEPS:-0,50,100,250,500}"
MIX_RATIO="${MIX_RATIO:-0.2}"
SEED="${SEED:-0}"
MODES="${ABSORB_MODES:-target_only,spynet,geometry}"

missing=0
for path in "$CHECKPOINT" "$PROMPT_JSON" "$SKYFALL_PY"; do
  if [[ ! -e "$path" ]]; then
    echo "missing: $path" >&2
    missing=1
  fi
done
IFS=',' read -r -a MODE_LIST <<< "$MODES"
for mode in "${MODE_LIST[@]}"; do
  refined="$ALIGN/$mode/refined.png"
  if [[ ! -f "$refined" ]]; then
    echo "missing: $refined" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "Absorption inputs are incomplete; refusing to launch." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"

for mode in "${MODE_LIST[@]}"; do
  out_dir="$OUT_ROOT/$mode"
  curve="$out_dir/zoom_2x/absorption_curve.json"
  if [[ -f "$curve" && "${FORCE:-0}" != "1" ]]; then
    echo "skip completed $mode ($curve exists; FORCE=1 to rerun)"
    continue
  fi
  if [[ "${FORCE:-0}" == "1" ]]; then
    rm -rf "$out_dir"
  elif [[ -d "$out_dir" && -n "$(ls -A "$out_dir" 2>/dev/null)" ]]; then
    echo "partial output exists in $out_dir; pass FORCE=1 to replace." >&2
    exit 1
  fi
  mkdir -p "$out_dir"
  echo "=== absorb $mode: ${STEPS} steps, mix=${MIX_RATIO}, eval=${EVAL_STEPS} ==="
  "$SKYFALL_PY" train_zoom_gen.py \
    --start_checkpoint "$CHECKPOINT" \
    --output_dir "$out_dir" \
    --view_index 0 \
    --roi_center_x 0.592 \
    --roi_center_y 0.53 \
    --roi_width 0.1 \
    --roi_height 0.1 \
    --zoom_factors 2 \
    --sr_scale 1.0 \
    --supervision_mode original \
    --steps_per_level "$STEPS" \
    --mix_ratio "$MIX_RATIO" \
    --eval_steps "$EVAL_STEPS" \
    --seed "$SEED" \
    --refine_backend reuse \
    --refined_image "$ALIGN/$mode/refined.png" \
    --prompt_json "$PROMPT_JSON" \
    --skip_geometry \
    --no_generation_cache
done

"$SKYFALL_PY" scripts/summarize_dloral_absorb.py --root "$OUT_ROOT"
echo "Absorption runs complete. Summary in $OUT_ROOT"

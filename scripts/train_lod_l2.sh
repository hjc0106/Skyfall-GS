#!/usr/bin/env bash
# Small 4x L2 check on frozen geometry L1. SpyNet 2x is a baseline, not a 4x rerun.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="${START_CHECKPOINT:-$ROOT/skyfall-gs_exp/stage1/JAX_068/chkpnt30000.pth}"
L1_CKPT="${L1_CHECKPOINT:-$ROOT/skyfall-gs_exp/lod_l1_absorb_2x/geometry/l1_final.lod.pt}"
OUT="${L2_OUTPUT_DIR:-$ROOT/skyfall-gs_exp/lod_l2_geometry_4x}"
PROMPT_JSON="${DLORAL_PROMPT_JSON:-$ROOT/skyfall-gs_exp/zoom_gen_dloral_spynet_2x/zoom_2x/prompt.json}"
GZ_ROOT="${GAUSSIANZOOM_ROOT:-$ROOT/vendor/gaussianzoom_distill}"
SKYFALL_PY="${SKYFALL_PYTHON:-$HOME/miniconda3/envs/skyfall-gs/bin/python}"
DLORAL_PY="${DLORAL_PYTHON:-$HOME/miniconda3/envs/dloral/bin/python}"
GPU="${ABSORB_GPU:-2}"
WEIGHT_ROOT="${DLORAL_WEIGHT_ROOT:-$ROOT/weights/dloral}"
SPYNET_2X="${SPYNET_2X_DIR:-$ROOT/skyfall-gs_exp/lod_l1_absorb_2x/spynet}"
SHORT_STEPS="${SHORT_STEPS:-50}"
STEPS="${STEPS_PER_LEVEL:-500}"
SEED="${SEED:-0}"
MIX_RATIO="${MIX_RATIO:-0.2}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
export DLORAL_WEIGHT_ROOT="$WEIGHT_ROOT"

if [[ ! -f "$L1_CKPT" ]]; then
  echo "missing frozen L1: $L1_CKPT" >&2
  exit 1
fi
if [[ ! -f "$PROMPT_JSON" ]]; then
  echo "missing prompt: $PROMPT_JSON" >&2
  exit 1
fi

mkdir -p "$OUT"

if [[ ! -f "$OUT/refined.png" || "${FORCE_SUPERVISION:-0}" == "1" ]]; then
  echo "=== 4x supervision from frozen L0+L1 ==="
  "$SKYFALL_PY" scripts/prepare_lod_l2_supervision.py \
    --start_checkpoint "$CHECKPOINT" \
    --l1_checkpoint "$L1_CKPT" \
    --output_dir "$OUT" \
    --prompt_json "$PROMPT_JSON" \
    --gz_root "$GZ_ROOT" \
    --view_index 0 \
    --roi_center_x "${ROI_CENTER_X:-0.592}" --roi_center_y "${ROI_CENTER_Y:-0.53}" \
    --roi_width "${ROI_WIDTH:-0.1}" --roi_height "${ROI_HEIGHT:-0.1}" \
    --zoom_factor 4 --step_scale 2 \
    --run_dloral \
    --dloral_python "$DLORAL_PY" \
    --dloral_device cuda:0 \
    --dloral_root "$ROOT/submodules/DLoRAL" \
    --sd_path "$WEIGHT_ROOT/stable-diffusion-2-1-base" \
    --ckpt "$WEIGHT_ROOT/model_enhanced.pkl" \
    --spynet "$WEIGHT_ROOT/spynet_20210409-c6c1bd09.pth" \
    --seed "$SEED"
fi

if [[ ! -f "$OUT/refined.png" ]]; then
  echo "4x refined.png was not written" >&2
  exit 1
fi

if [[ ! -f "$OUT/steps/0050/metrics.json" || "${FORCE_SHORT:-0}" == "1" ]]; then
  echo "=== L2 short run: ${SHORT_STEPS} steps ==="
  "$SKYFALL_PY" scripts/train_lod_l2.py \
    --start_checkpoint "$CHECKPOINT" \
    --l1_checkpoint "$L1_CKPT" \
    --output_dir "$OUT" \
    --refined_image "$OUT/refined.png" \
    --gz_root "$GZ_ROOT" \
    --view_index 0 \
    --roi_center_x "${ROI_CENTER_X:-0.592}" --roi_center_y "${ROI_CENTER_Y:-0.53}" \
    --roi_width "${ROI_WIDTH:-0.1}" --roi_height "${ROI_HEIGHT:-0.1}" \
    --zoom_factor 4 --parent_zoom_factor 2 --step_scale 2 \
    --steps "$SHORT_STEPS" --eval_steps "0,$SHORT_STEPS" \
    --mix_ratio "$MIX_RATIO" --seed "$SEED" \
    --skip_cross_view
fi

export L2_OUT="$OUT"
TRAIN_GT="$("$SKYFALL_PY" -c "import json,os; print(json.load(open(os.environ['L2_OUT']+'/steps/0050/metrics.json'))['train_views_mean']['l1_to_gt_mean'])")"
echo "1x train GT after ${SHORT_STEPS} steps: $TRAIN_GT"
"$SKYFALL_PY" -c "import sys; gt=float(sys.argv[1]); sys.exit(0 if gt < 0.03 else 1)" "$TRAIN_GT"

if [[ ! -f "$OUT/steps/0500/metrics.json" || "${FORCE_TRAIN:-0}" == "1" ]]; then
  echo "=== L2 continue to ${STEPS} steps ==="
  "$SKYFALL_PY" scripts/train_lod_l2.py \
    --start_checkpoint "$CHECKPOINT" \
    --l1_checkpoint "$L1_CKPT" \
    --resume "$OUT/l2_final.lod.pt" \
    --start_step "$SHORT_STEPS" \
    --output_dir "$OUT" \
    --refined_image "$OUT/refined.png" \
    --gz_root "$GZ_ROOT" \
    --view_index 0 \
    --roi_center_x "${ROI_CENTER_X:-0.592}" --roi_center_y "${ROI_CENTER_Y:-0.53}" \
    --roi_width "${ROI_WIDTH:-0.1}" --roi_height "${ROI_HEIGHT:-0.1}" \
    --zoom_factor 4 --parent_zoom_factor 2 --step_scale 2 \
    --steps "$STEPS" --eval_steps "100,250,500" \
    --mix_ratio "$MIX_RATIO" --seed "$SEED"
fi

"$SKYFALL_PY" scripts/summarize_lod_l2.py --root "$OUT" --spynet_2x "$SPYNET_2X"
echo "L2 4x check complete. Summary in $OUT/lod_l2_summary.json"

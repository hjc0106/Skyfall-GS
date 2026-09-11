#!/usr/bin/env bash
# Native DLoRAL dual-view SpyNet baseline: one checkpoint, one ROI, 2x only.
# Reuses the archived Qwen3-VL prompt. Does not download weights.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHT_ROOT="${DLORAL_WEIGHT_ROOT:-$ROOT/weights/dloral}"
SD_DIR="${DLORAL_SD_PATH:-$WEIGHT_ROOT/stable-diffusion-2-1-base}"
CKPT="${DLORAL_CKPT:-$WEIGHT_ROOT/model.pkl}"
SPYNET="${DLORAL_SPYNET:-$WEIGHT_ROOT/spynet_20210409-c6c1bd09.pth}"
PY_DLORAL="${DLORAL_PYTHON:-$HOME/miniconda3/envs/dloral/bin/python}"
OUT="${DLORAL_OUTPUT_DIR:-$ROOT/skyfall-gs_exp/zoom_gen_dloral_spynet_2x}"
PROMPT_JSON="${DLORAL_PROMPT_JSON:-$ROOT/skyfall-gs_exp/zoom_gen_20260908/skyfall_zoom_gen_sr_vlm_geometry_20260908/zoom_2x/prompt.json}"
CHECKPOINT="${START_CHECKPOINT:-$ROOT/skyfall-gs_exp/stage1/JAX_068/chkpnt30000.pth}"

missing=0
for path in "$SD_DIR/model_index.json" "$CKPT" "$SPYNET" "$PY_DLORAL" "$PROMPT_JSON" "$CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "missing: $path" >&2
    missing=1
  fi
done
if [[ ! -f "$SD_DIR/unet/diffusion_pytorch_model.fp16.bin" && ! -f "$SD_DIR/unet/diffusion_pytorch_model.safetensors" && ! -f "$SD_DIR/unet/diffusion_pytorch_model.bin" ]]; then
  echo "missing: SD unet weights under $SD_DIR/unet" >&2
  missing=1
fi
if [[ "$missing" -ne 0 ]]; then
  echo "DLoRAL weights/env are incomplete; refusing to launch (no implicit download)." >&2
  exit 1
fi

cd "$ROOT"
"${SKYFALL_PYTHON:-$HOME/miniconda3/envs/skyfall-gs/bin/python}" train_zoom_gen.py \
  --start_checkpoint "$CHECKPOINT" \
  --output_dir "$OUT" \
  --view_index 0 \
  --roi_center_x 0.592 \
  --roi_center_y 0.53 \
  --roi_width 0.1 \
  --roi_height 0.1 \
  --zoom_factors 2 \
  --sr_scale 1.0 \
  --supervision_mode original \
  --steps_per_level "${STEPS_PER_LEVEL:-5}" \
  --mix_ratio 0.0 \
  --refine_backend dloral \
  --dloral_alignment spynet \
  --dloral_root "$ROOT/submodules/DLoRAL" \
  --dloral_python "$PY_DLORAL" \
  --dloral_sd_path "$SD_DIR" \
  --dloral_ckpt "$CKPT" \
  --dloral_spynet "$SPYNET" \
  --dloral_device "${DLORAL_DEVICE:-cuda:2}" \
  --dloral_stages 1 \
  --dloral_process_size 512 \
  --dloral_upscale 1 \
  --dloral_latent_tiled_size "${DLORAL_LATENT_TILED_SIZE:-96}" \
  --prompt_json "$PROMPT_JSON" \
  --geometry_neighbor_count 2 \
  --geometry_pool_size 8

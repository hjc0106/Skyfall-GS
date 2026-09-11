#!/usr/bin/env bash
# 2048 tile=96 SpyNet / geometry / target_only. Same prompt, weights, seed.
# Requires dual-depth pair from export_dloral_geometry_pair.py.
# Does not train 3DGS. Inspect seams and occlusions; L1 is not a ranking.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHT_ROOT="${DLORAL_WEIGHT_ROOT:-$ROOT/weights/dloral}"
export DLORAL_PYTHON="${DLORAL_PYTHON:-$HOME/miniconda3/envs/dloral/bin/python}"
export DLORAL_DEVICE="${DLORAL_DEVICE:-cuda:2}"
PAIR_DIR="${DLORAL_GEOMETRY_DIR:-$ROOT/skyfall-gs_exp/dloral_geometry_pair}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
SKYFALL_PY="${SKYFALL_PYTHON:-$HOME/miniconda3/envs/skyfall-gs/bin/python}"

if [[ ! -f "$PAIR_DIR/source_to_target_flow.npy" ]]; then
  "$SKYFALL_PY" scripts/export_dloral_geometry_pair.py \
    --baseline_dir "$ROOT/skyfall-gs_exp/zoom_gen_dloral_spynet_2x/zoom_2x" \
    --output_dir "$PAIR_DIR" \
    --start_checkpoint "$ROOT/skyfall-gs_exp/stage1/JAX_068/chkpnt30000.pth" \
    --device "$DLORAL_DEVICE"
fi

"$SKYFALL_PY" scripts/compare_dloral_align_512.py \
  --baseline_dir "$ROOT/skyfall-gs_exp/zoom_gen_dloral_spynet_2x/zoom_2x" \
  --output_dir "${DLORAL_OUTPUT_DIR:-$ROOT/skyfall-gs_exp/zoom_gen_dloral_align_2048}" \
  --geometry_dir "$PAIR_DIR" \
  --size 2048 \
  --device "$DLORAL_DEVICE" \
  --dloral_python "$DLORAL_PYTHON" \
  --sd_path "${DLORAL_SD_PATH:-$WEIGHT_ROOT/stable-diffusion-2-1-base}" \
  --ckpt "${DLORAL_CKPT:-$WEIGHT_ROOT/model.pkl}" \
  --spynet "${DLORAL_SPYNET:-$WEIGHT_ROOT/spynet_20210409-c6c1bd09.pth}" \
  --dloral_root "$ROOT/submodules/DLoRAL"

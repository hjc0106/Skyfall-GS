#!/usr/bin/env bash
# Create an isolated DLoRAL inference environment. Do not install into skyfall-gs.
# Weights are not downloaded here; place them under DLORAL_WEIGHT_ROOT yourself.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN="$ROOT/submodules/DLoRAL.pin"
ENV_NAME="${DLORAL_ENV_NAME:-dloral}"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
WEIGHT_ROOT="${DLORAL_WEIGHT_ROOT:-$ROOT/weights/dloral}"

echo "DLoRAL pin:"
cat "$PIN"
echo

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "conda not found at $CONDA_BIN" >&2
  exit 1
fi

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda env $ENV_NAME (python 3.10)"
  "$CONDA_BIN" create -y -n "$ENV_NAME" python=3.10
else
  echo "Conda env $ENV_NAME already exists"
fi

PY="$HOME/miniconda3/envs/$ENV_NAME/bin/python"
PIP="$HOME/miniconda3/envs/$ENV_NAME/bin/pip"

echo "Installing torch 2.1.2+cu121 (mmcv wheels exist for this combo; official 2.0.1 does not on CUDA 12.8)"
"$PIP" install --upgrade pip wheel
"$PIP" install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

echo "Installing mmengine + mmcv CUDA ops"
"$PIP" install mmengine
"$PIP" install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

echo "Installing DLoRAL inference Python deps (no RAM/DAPE, no jax/basicsr training extras, no xformers)"
"$PIP" install \
  'numpy==1.24.4' \
  'diffusers==0.25.0' \
  'transformers==4.28.1' \
  'huggingface_hub==0.25.0' \
  'peft==0.9.0' \
  'einops==0.7.0' \
  'Pillow' \
  'PyYAML' \
  'opencv-python==4.9.0.80' \
  'accelerate' \
  'safetensors'

echo "Verifying mmcv CUDA ops and DLoRAL imports"
"$PY" - <<PY
import sys
sys.path.insert(0, "$ROOT/submodules/DLoRAL")
import numpy, torch
from mmcv.ops import ModulatedDeformConv2d
from src.cross_frame_retrieval import cfr_main
from src.DLoRAL_model import Generator_eval
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("mmcv ModulatedDeformConv2d", ModulatedDeformConv2d)
print("DLoRAL modules import OK")
PY

SPYNET="$WEIGHT_ROOT/spynet_20210409-c6c1bd09.pth"
SD_DIR="$WEIGHT_ROOT/stable-diffusion-2-1-base"
CKPT="$WEIGHT_ROOT/model.pkl"

echo
echo "Environment: $PY"
echo "Weights are not downloaded by this script. Expected paths:"
echo "  SpyNet : $SPYNET"
echo "  SD 2.1 : $SD_DIR  (needs model_index.json; official DLoRAL uses yujingsun/stable-diffusion-2-1-base)"
echo "  DLoRAL : $CKPT    (improved: 1C2TLERta3a-PkoMpqQhHoM_S54pNEASO)"
echo
echo "Use with train_zoom_gen.py:"
echo "  --refine_backend dloral"
echo "  --dloral_root $ROOT/submodules/DLoRAL"
echo "  --dloral_python $PY"
echo "  --dloral_sd_path $SD_DIR"
echo "  --dloral_ckpt $CKPT"
echo "  --dloral_spynet $SPYNET"

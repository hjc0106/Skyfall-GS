#!/usr/bin/env bash
# Install official RaDe-GS diff_gaussian_rasterization into skyfall-gs.
# Does not replace Skyfall's existing diff_gauss package.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN="$ROOT/submodules/RaDe-GS.pin"
DEST="${RADEGS_RASTERIZER:-$ROOT/submodules/diff-gaussian-rasterization}"
PY="${SKYFALL_PYTHON:-$HOME/miniconda3/envs/skyfall-gs/bin/python}"
PIP="${SKYFALL_PIP:-$HOME/miniconda3/envs/skyfall-gs/bin/pip}"
REPO_URL="${RADEGS_URL:-https://github.com/HKUST-SAIL/RaDe-GS.git}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

echo "RaDe-GS pin:"
cat "$PIN"
echo

if [[ ! -x "$PY" ]]; then
  echo "missing python: $PY" >&2
  exit 1
fi
if [[ ! -d "$CUDA_HOME" ]]; then
  echo "missing CUDA_HOME: $CUDA_HOME" >&2
  exit 1
fi

NEED_CLONE=1
if [[ -f "$DEST/setup.py" && -d "$DEST/diff_gaussian_rasterization" ]]; then
  NEED_CLONE=0
fi
if [[ "$NEED_CLONE" -eq 1 ]]; then
  tmp="$(mktemp -d)"
  echo "Sparse-cloning $REPO_URL into $DEST"
  git init "$tmp/RaDe-GS"
  git -C "$tmp/RaDe-GS" remote add origin "$REPO_URL"
  RADE_COMMIT="$(sed -n 's/^commit=//p' "$PIN")"
  git -C "$tmp/RaDe-GS" fetch --depth 1 origin "$RADE_COMMIT"
  git -C "$tmp/RaDe-GS" sparse-checkout init --cone
  git -C "$tmp/RaDe-GS" checkout --detach FETCH_HEAD
  git -C "$tmp/RaDe-GS" sparse-checkout set submodules/diff-gaussian-rasterization
  rm -rf "$DEST"
  mkdir -p "$(dirname "$DEST")"
  mv "$tmp/RaDe-GS/submodules/diff-gaussian-rasterization" "$DEST"
  git -C "$tmp/RaDe-GS" rev-parse HEAD > "$DEST/.rade_gs_commit"
  rm -rf "$tmp"
fi

GLM_HPP="$DEST/third_party/glm/glm/glm.hpp"
if [[ ! -f "$GLM_HPP" ]]; then
  GLM_FALLBACK="$ROOT/submodules/diff-gaussian-rasterization-depth/third_party/glm"
  mkdir -p "$DEST/third_party"
  if [[ -f "$GLM_FALLBACK/glm/glm.hpp" ]]; then
    echo "Copying glm headers from $GLM_FALLBACK (sparse clone skips the glm submodule)"
    rsync -a --exclude .git "$GLM_FALLBACK/" "$DEST/third_party/glm/"
  else
    echo "Cloning g-truc/glm into $DEST/third_party/glm"
    git clone --depth 1 https://github.com/g-truc/glm.git "$DEST/third_party/glm"
  fi
fi
if [[ ! -f "$GLM_HPP" ]]; then
  echo "missing glm headers: $GLM_HPP" >&2
  exit 1
fi
rm -rf "$DEST/build" "$DEST/"*.egg-info

export CUDA_HOME
export PATH="$CUDA_HOME/bin:${PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

echo "Building diff_gaussian_rasterization with $PY (torch/cuda from this env)"
"$PIP" install --no-build-isolation "$DEST"

"$PY" - <<'PY'
import diff_gauss
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
fields = getattr(GaussianRasterizationSettings, "_fields", ())
if "require_depth" not in fields:
    raise SystemExit(f"installed rasterizer is not RaDe-GS: fields={fields}")
print("diff_gauss still OK", diff_gauss)
print("diff_gaussian_rasterization OK", GaussianRasterizer, "fields", fields)
PY
echo "RaDe-GS rasterizer is installed. Skyfall diff_gauss is unchanged."

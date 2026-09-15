#!/usr/bin/env bash
# Local all-scene evaluation queue (owner: DatasetEvaluator, local GPU0 only).
#
# Long-running consumer: waits for any <scene>/<stage> whose ArchiveCurator
# archive_status.json says status=verified and evaluates it, publishing
# evaluation_status.json (status=running + model_load_verified=true as soon as the
# model loads and the first held-out view renders, then status=completed when the
# full metric set and missing-GT notes are in place).
#
# Only ONE queue may run at a time (flock on <archive_root>/.eval_queue.lock).
#
#   ./scripts/run_eval_queue.sh                 # wait for verified archives, evaluate all 12 scenes
#   ./scripts/run_eval_queue.sh --queue-once    # single pass over currently verified archives
#   ./scripts/run_eval_queue.sh --exit-when-complete
#
# Existing JAX_068 stage1+stage2 are evaluated immediately; the rest arrive as
# remote training finishes and ArchiveCurator verifies each transfer.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${EVAL_PYTHON:-/home/guishe/anaconda3/envs/skyfall-v5/bin/python}"
MANIFEST="${EVAL_MANIFEST:-/dataset/Skyfall-GS/experiments/stage2_gaussianzoom_20260913/pipeline_manifest.json}"
GPU="${EVAL_GPU:-0}"
LOG="${EVAL_LOG:-/dataset/Skyfall-GS/experiments/stage2_gaussianzoom_20260913/eval_queue.log}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
# Keep the queue from competing with the remote-training-linked CPU work.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$(dirname "$LOG")"

exec "$PY" -u scripts/evaluate_dataset.py \
  --manifest "$MANIFEST" \
  --queue \
  --all-scenes \
  --stages stage1 stage2 \
  --official \
  --skip-completed \
  --poll-interval "${EVAL_POLL_INTERVAL:-120}" \
  "$@" 2>&1 | tee -a "$LOG"

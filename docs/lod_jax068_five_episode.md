# JAX_068 L0-start five-episode LoD experiment

This experiment starts from the Stage 1 L0 checkpoint at 30K, creates an empty
L1, freezes L0 and appearance, and trains the same L1 through five consecutive
episodes. It is independent of the earlier six-view continuation archive.

## Locked protocol

- Episode `(elevation, radius)`: `(85°, 300)`, `(75°, 275)`, `(65°, 250)`,
  `(55°, 225)`, `(45°, 200)`.
- Per episode: 9 look-at centers × 6 azimuths × 2 generated samples = 108
  supervision images.
- Each episode trains for 5,000 local steps using only its current 108-image
  pool.
- L0, appearance MLP, and appearance embeddings remain frozen.
- L1 uses the full learning rates from the original two-scale entry, a shared
  450K-point cap, one Adam state across episodes, and cumulative densification
  through global step 20K.
- The six held-out azimuths are used only for evaluation.

The machine-readable contract is in `lod/episodes_l0_108.py`. The executable
pipeline is `scripts/run_lod_jax068_c_episodes_l0_108.py`.

## Required paths

Set these variables before using the shell wrapper:

```bash
export JAX068_DATASET_DIR=/path/to/datasets_JAX/JAX_068
export DLORAL_WEIGHT_ROOT=/path/to/dloral/weights
export VLM_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
export FLOWEDIT_MODEL_PATH=/path/to/FLUX.1-dev

# Optional when the three stages use different environments.
export SKYFALL_PYTHON=/path/to/skyfall/python
export DLORAL_PYTHON=/path/to/dloral/python
export VLM_PYTHON=/path/to/vlm/python
```

The Stage 1 checkpoint defaults to
`skyfall-gs_exp/stage1/JAX_068/chkpnt30000.pth`. Override it with
`START_CHECKPOINT` when needed. Generated experiment data stays under
`skyfall-gs_exp/`, which is ignored by Git.

## Run and resume

Run the independent integration probe first:

```bash
ABSORB_GPU=0 bash scripts/run_lod_jax068_c_episodes_l0_108.sh probe
```

After `probe/PROBE.json` reports success, run all five episodes:

```bash
ABSORB_GPU=0 bash scripts/run_lod_jax068_c_episodes_l0_108.sh all
```

Re-running the same command resumes from validated supervision and local
checkpoints. Reuse is refused when the parent checkpoint hash or locked
protocol changes. Probe checkpoints are never used as the formal start.

Useful recovery phases are `supervise`, `flowedit`, `train`, `evaluate`,
`episode`, and `report`; pass `--episode N` after the phase:

```bash
ABSORB_GPU=0 bash scripts/run_lod_jax068_c_episodes_l0_108.sh evaluate \
  --episode 3
```

Do not point `C_EPISODES_L0_108_DIR` into
`skyfall-gs_exp/lod_jax068_c_episodes_l1`; the runner rejects that completed
archive.

## Verification

The focused CPU checks are:

```bash
python -m pytest -q \
  tests/test_lod_episodes_l0_108.py \
  tests/test_lod_episodes_l0_108_runner.py
```

The probe additionally validates empty-L1 equivalence, frozen state, nonzero
L1 gradients and updates, interrupted-run resume, densification after resume,
dual-sample seeds, and current-episode-only sampling.

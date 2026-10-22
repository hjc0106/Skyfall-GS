# B/C Stage 1 DSM migration plan

Status: implementation delivered in the independent branch; see stage1_dsm_usage.md.
Real JAX_068 coordinate acceptance remains open. No main-branch merge performed.
Target base: Skyfall-GS main, 22134e125fe9c4ee21d3bfcc105ec89275b26dcc.
Implementation worktree: existing independent codex/stage1-dsm-prior branch.
Source: /home/hongjiacheng/codes/ArtiFixer/thirdparty/Skyfall-GS,
plus /home/hongjiacheng/codes/ArtiFixer/scripts/prepare_dsm_prior.py.
The dev-gszoom-v2 checkout, its uncommitted changes, and source ArtiFixer files
remain untouched. Integration into main follows a reviewable branch diff.

## User-facing behavior

The surface constraint switch defaults to enabled. With a supplied DSM prior,
Stage 1 defaults to the source B/C mixed initializer. Without a DSM prior,
Stage 1 uses the original SfM path and the DSM loss is inactive.

| Inputs | Initialization | Effective surface loss | Experiment |
|---|---|---|---|
| No DSM prior | SfM | inactive | A / existing Stage 1 |
| --dsm_prior PATH | DSM mixed | enabled by default | C |
| --dsm_prior PATH --no_dsm_surface | identical DSM mixed | disabled | B |

Support an explicit --dsm_surface enable flag as well. Implement the two
switches with one boolean destination and a real negative action; default-true
store_true alone is insufficient. Conflicting switches must be rejected.
Retain/translate the source --dsm_init argument for compatibility; record the
resolved init mode separately from the surface flag. A nonempty invalid prior
path fails before training, even when the surface loss is disabled.

The default positive surface coefficient is 0.05, matching source C.
--no_dsm_surface takes precedence over the configured coefficient and skips
surface sampling/computation entirely. An enabled flag with a zero coefficient
must be reported as effectively inactive, rather than claimed to be C.
Save requested and effective settings and active prior-point count.
Stage 2 does not implicitly inherit the new Stage 1 surface default.

## Scope: preserve source B/C behavior

Migrate the existing mixed initializer, not a new DSM reconstruction method:
source replace_budgeted selection, near-surface replacement, floater removal,
point budget, nearest-SfM color, DSM-normal-aligned short axes, tangent/normal
scales, holdout masking and child confidence decay. Keep existing registered
NPZ grids authoritative; sampling spacing derives from grid GSD and sample_step.

Preserve the source surface term for reproducibility:

    h, n, c = sample_registered_DSM(current_xy)
    r = dot(xyz - [x, y, h], detach(n))
    w = detach(c) * detach(opacity) * prior_confidence
    L_surface = sum(w * log1p((r / sigma)^2)) / max(sum(w), eps)
    L_total = L_existing_stage1 + lambda_surface * schedule * L_surface

Defaults: sigma=0.5 in the existing registered metric frame, lambda=0.05;
source schedule warmup=500, hold=5000, decay=21000, late_scale=0.2.
Smoke can explicitly override warmup to 20. Source dtype handling around
opacity resets must be retained. Relative/normal losses and later D--I or
appearance-only experiments are outside this B/C migration. Normals remain
needed for initialization and the surface distance itself.

The earlier proposed dead-zone Huber, visibility-median color, strict 2 m
initializer and hard 5-tau cutoff are different algorithms. They are not
silently substituted for source B/C in this migration.

## Changes by dependency order

1. Add a self-contained utils/dsm_prior.py with NPZ validation, DSM sampling,
   mixed initialization and the surface loss/schedule. Migrate preparation
   helpers without imports back into ArtiFixer or absolute data paths. Accept
   old registered NPZ/registration.json; add schema, source hash and coordinate
   metadata to newly produced artifacts. Keep datasets/weights out of Git.
2. Add parameters, the default-enabled surface switch and resolved config
   logging. Validate shapes, finite heights/normals, valid/holdout masks,
   positive scales, GSD and point limits. Disabled surface mode still requires
   the prior for the mixed initializer.
3. Extend SceneInfo and Satellite reader to return DSM initialization metadata;
   pass it into create_from_pcd. Only generate source points3D.ply when missing;
   write mixed initialization artifacts into the experiment output directory.
4. Migrate Gaussian prior confidence/anchor tensors and anisotropic init.
   Update clone/split/prune in exactly matching point order and assert lengths.
   Store confidence decay and prior identity, not only confidence tensor values.
5. Connect the surface term only in Stage 1 before backward. Load MoGe/FlowEdit
   imports and model instances only at actual use sites; pseudo-depth=0 loads
   no MoGe. Existing RGB, appearance, opacity and pseudo-depth configuration
   remain independently configurable.
6. Add versioned checkpoint adapters for original main checkpoints, source
   ArtiFixer DSM checkpoints and the new format. Save prior identity/state,
   coordinates, config, optimizer/RNG state and densification/loss schedules.
   New-format resumes preserve schedules; verify old readers/render/export
   integration through central loaders rather than duplicating tuple unpacking.
7. Migrate the geometry evaluator and add a structured A/B/C runner. Paths,
   device, seed and output directory must be configurable. Do not import the
   source experimental launcher unchanged: it disables opacity and MoGe and
   uses screening-specific settings.

## Coordinate and scientific validation

The existing registered prior uses scene.loc for horizontal origin and an
SfM-derived robust height offset (stored dz approximately -46.9727 m). Treat
this as an artifact value with provenance, never a hardcoded scene constant.
Source inputs match the currently inspected JAX_068 point/camera files.

Validate NPZ-to-world/world-to-NPZ roundtrip and actual training Camera
projection before training; record any R/T/scale normalization applied by the
loader. Do not normalize or fit a new similarity merely because camera centers
are far away: the original off-center Camera projection already puts most SfM
points in frame. A new coordinate scheme would be a separate change.

The earlier strict requirement for measured track reprojection is still an
open acceptance item: current COLMAP 2D observations have not been located;
the RPC crop observations index another reconstruction. Existing registration
and successful historical training do not establish that check. Migrate/test
code and compare saved artifacts independently; do not claim the full strict
coordinate/smoke acceptance gate passed while this item is unresolved.

Reuse the geometry evaluator's original center-raster metric as a compatibility
report. Also export valid coverage and a common-predicted-grid A/B/C comparison.
Label shared-prior/reference evaluation controlled_same_dsm. The spatial holdout
is not independent ground truth, and source global height registration used
the full DSM. Do not silently mix these metrics with rendered-surface metrics.

## Validation and deliverables

CPU tests:
- default C, negative-switch B, no-prior A and invalid-path/error semantics;
- source/migrated initializer outputs on fixed data and seed;
- source/migrated surface values and gradients on identical tensors;
- disabled surface contributes no term or gradient;
- prior alignment through clone, split and prune; opacity-reset dtype;
- old/new checkpoint loading and schedule-preserving new resume.

Functional smoke (only after required coordinate checks pass):
JAX_068 A/B/C, identical seeds/camera split, resolution=4, about 200 steps,
explicit initial-point limit, densify from 50 every 50, opacity reset at 100,
checkpoint/resume at 100, surface warmup=20. MoGe disabled for all three;
normal/relative losses absent. Ensure actual topology/reset events happen,
finite losses/gradients and a successful resumed render. Log current inactive
losses and constrained-point counts so a no-op C cannot pass unnoticed.

Separate configs:
- source_bc_replay: source B/C schedule/initializer and screening settings;
- stage1_dsm: main's normal Stage 1 losses plus optional DSM surface term.
Do not equate the latter's numerical results with historical source B/C.

Deliver a branch diff, portable CLI examples, configuration presets,
source-parity tests and reports, smoke_report.json and metric exports.
Full 30k/multi-seed/point-budget-matched experiments remain for other servers.
No merge into main is part of this planning step.

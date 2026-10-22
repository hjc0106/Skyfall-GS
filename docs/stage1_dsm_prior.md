# Stage 1 DSM prior implementation status

The B/C migration has now been implemented in the independent
`codex/stage1-dsm-prior` worktree based on main `22134e1`.
See [usage and verification](stage1_dsm_usage.md) and the
[migration plan](stage1_dsm_migration_plan.md).

Implemented: registered-prior builder, source B/C mixed initialization, default-on
surface switch, source surface math/schedule, Gaussian prior inheritance through
clone/split/prune, PLY prior state, legacy/versioned checkpoint support,
schedule/RNG-preserving new resume, lazy MoGe/FlowEdit, limited RGB/geometry
assessment, common-grid comparison, strict coordinate validator and A/B/C runner.

Coordinate acceptance for real JAX_068 remains open. Its current reader is
identity (no R/T). Actual Camera projection places most SfM points in images;
large camera distances alone do not establish misregistration. But genuine
matching 2D observations for points3D.txt have not been located. RPC crop JSONs
reference a separate ECEF reconstruction and cannot replace those observations.
The real A/B/C smoke remains blocked and has not been started.

Known registered-prior frame restriction: additional R/T normalization requires
an explicitly transformed/supported prior, rather than an implicit grid rotation.
The local source JAX_068 follows the supported identity reader path.

# PRF patch field protocol — schema version 2

## Patch geometry

A patch is a local planar support region derived from its member Gaussians. It is
not a fixed-radius disk.

- `center` is the opacity-weighted local center in world metres.
- `normal`, `tangent_1`, and `tangent_2` are an orthonormal world-space basis.
- Member Gaussian centres are projected to the tangent plane.
- `hull_uv` stores the cyclic 2D convex-hull vertices in metres.
- World hull vertex `q_k` is reconstructed as
  `center + hull_uv[k,0] * tangent_1 + hull_uv[k,1] * tangent_2`.
- `patch_area` is the shoelace area of that hull in square metres.
- `patch_radius` is the median tangent-plane distance of member Gaussians from
  the patch centre. It is a support statistic, not display geometry.

## NPZ arrays

`physical_resolution_field.npz` is a non-pickle NumPy archive. Patch-major
arrays use `N = len(patch_id)`.

| Key | Shape | Meaning |
| --- | --- | --- |
| `schema_version` | scalar | Integer `2` |
| `patch_id` | `[N]` | Unique patch identifier |
| `center` | `[N,3]` | World-space centre, metres |
| `normal` | `[N,3]` | Unit surface normal |
| `tangent_1`, `tangent_2` | `[N,3]` | Unit tangent basis |
| `patch_radius` | `[N]` | Median support radius, metres |
| `patch_area` | `[N]` | Convex-hull area, m² |
| `num_gaussians` | `[N]` | Number of member Gaussians |
| `num_visible_real_views` | `[N]` | Frustum-visible training views |
| `best_pair_angle_deg` | `[N]` | Best camera-pair angle |
| `best_camera_a`, `best_camera_b` | `[N]` | Best pair view IDs; empty if absent |
| `quality_flags` | `[N]` | `|`-separated diagnostic flags |
| `R_obs`, `R_kernel`, `R_spacing`, `R_phys` | `[N]` | Metres per equivalent pixel |
| `bottleneck` | `[N]` | `KERNEL`, `SPACING`, `OBS`, or `INVALID` |
| `member_offsets` | `[N+1]` | CSR offsets into `member_gaussian_ids` |
| `member_gaussian_ids` | `[M]` | Source Gaussian indices |
| `hull_offsets` | `[N+1]` | CSR offsets into `hull_uv` |
| `hull_uv` | `[H,2]` | Tangent-plane convex-hull vertices, metres |

For patch `i`, members are
`member_gaussian_ids[member_offsets[i]:member_offsets[i+1]]` and the hull is
`hull_uv[hull_offsets[i]:hull_offsets[i+1]]`.

## Validity

`R_phys = max(R_obs, R_kernel, R_spacing)`. If `R_obs` is infinite because no
valid multi-view pair exists, `R_phys` is infinite and `bottleneck` must be
`INVALID`. An infinite observation is never classified as `OBS`.

Current visibility can be frustum/image-bound only, or training-view Gaussian
expected-depth occlusion. Depth occlusion is multi-state (`VISIBLE`,
`OCCLUDED`, `OUTSIDE_FRUSTUM`, `UNCERTAIN`, `INVALID_PROJECTION`); only
`VISIBLE` views enter the best camera pair. `UNCERTAIN` covers translucent
alpha and depth differences inside a tolerance that depends on Gaussian
scale, patch thickness, incidence, and float32 depth quantization. WorldAct
`RGB+D` is accumulated z (`sum w_i z_i`); PRF converts it to expected z
(`sum w_i z_i / alpha`) before the test. This is not a binary ground truth.

Patch neighborhoods may be split by sign-invariant normal clustering, then by
spatial connected components inside each cluster. One anchor can therefore
emit several candidate patches; near-duplicates are removed by centre
distance, unsigned normal agreement, and member IoU. `MIXED_NEIGHBORHOOD`
means the original KNN contained more than one orientation cluster. `BORDER`
is a hull-margin geometric-boundary flag, not `inliers / K < 0.5`.
`geometry_valid` is the trusted-decision subset (support, inlier fraction,
eigen ratio, hull, not severe border / low opacity). `NONPLANAR` patches may
remain for debug display but are not `geometry_valid`. Optional NPZ keys
`geometry_valid`, `geometry_confidence`, `plane_residual_m`, `support_count`,
`inlier_fraction`, and `eigen_ratio` are allowed extras; required keys stay
schema v2. `roof` / `road` / `facade` diagnostic labels are geometric
heuristics, not semantic classes.

## 3D viewer contract

The main 3D layer maps the four patch fields onto the original PLY Gaussian
centres with a surface-aware weighted interpolation over the eight nearest
patches. Tangent distance, normal distance, patch radius, and Gaussian/patch
normal agreement contribute to the weight. Non-finite values do not contribute;
the nearest patch determines whether `R_obs` or `R_phys` remains invalid.

The mapped original-Ply points and optional convex-hull surfaces share absolute
bins: `≤0.5`, `0.5–1`, `1–2`, `2–5`, `5–10`, and `>10` m/equiv.pixel. The viewer
can also restore SH0 RGB. Selecting a patch exposes its CSR members and projects
the exact hull into training-image previews. This is point-field rendering, not
3DGS rasterization.

`patches.bin` uses magic `PRF2`. `gaussian_context.bin` uses `GSF1`: point count,
float32 XYZ relative to `metadata.origin`, uint8 SH0 RGB, four uint8 resolution
bin arrays in `R_obs`, `R_kernel`, `R_spacing`, `R_phys` order, then uint8
bottleneck codes. Resolution code 254 is unmapped and retains source RGB; 255
is invalid. Bottleneck code 4 is unmapped. The viewer still accepts
legacy `GSC1`/`GSC2`/`GSC3` payloads. All binary numbers are little-endian.

The exported `*_prf_mapped.ply` stores the original Gaussian centres and normals,
standard RGB with a 68% `R_phys` overlay, pure `prf_red/green/blue`,
`source_red/green/blue`, all four float scalar fields, `bottleneck_code`,
`prf_valid`, and `prf_covered`.

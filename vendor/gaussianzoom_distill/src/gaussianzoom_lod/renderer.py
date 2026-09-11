"""RaDe-GS rendering with GaussianZoom Eq. (8)-(9) opacity modulation."""
import math

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def lod_weight(psi_current, psi_ref, step_scale):
    """Log-space tent, including zero contribution outside [1/s, s]."""
    if not math.isfinite(step_scale) or step_scale <= 1:
        raise ValueError('step_scale must be finite and > 1')
    # Reference coefficients are validated on model creation/load, not per frame.
    ratio = psi_current.clamp_min(torch.finfo(psi_current.dtype).tiny) / psi_ref
    return (1.0 - ratio.log().abs() / math.log(step_scale)).clamp_min(0.0)


def topology_weight(psi_current, psi_ref, step_scale, has_parent, has_child):
    """Supplement B.2 availability-aware log interpolation.

    Use main Eq.(9)'s plus sign on r<1 (Alg.B prints an inconsistent minus).
    Missing coarse/fine counterpart retains opacity on that side; w(1)=1.
    Inclusive boundary choices prevent isolated zero-opacity root/leaf frames.
    """
    tent = lod_weight(psi_current, psi_ref, step_scale)
    parent = has_parent.reshape(psi_ref.shape)
    child = has_child.reshape(psi_ref.shape)
    counterpart = torch.where(psi_current >= psi_ref, parent, child)
    return torch.where(counterpart, tent, torch.ones_like(tent))


def interval_weights(psi_current, psi_ref, stage_scales, data, layer_slices):
    """Partition actual creation-scale intervals, including skipped generations.

    Direct children sharing a parent and creation stage form one spatial cohort.
    Alternative cohorts use nonuniform log-scale hats evaluated at the SAME
    parent coordinate. Incoming mass propagates down the tree before each node
    hands it to its own children. Do not divide spatial siblings by their count.
    For an adjacent, equal-center single path this reduces to Eq.(9).
    """
    scales = torch.as_tensor(stage_scales, dtype=psi_current.dtype, device=psi_current.device)
    logs = scales.log()
    levels = data['node_levels']
    parents = data['parent_index']
    first = data['first_child_level']
    previous = data['previous_sibling_level']
    following = data['next_sibling_level']
    own = logs[levels][:, None]
    ratio = psi_current.clamp_min(torch.finfo(psi_current.dtype).tiny) / psi_ref
    coordinate = own - ratio.log()

    def ramp(value, low, high):
        return ((value-low)/(high-low)).clamp(0., 1.)

    has_child = (first >= 0)[:, None]
    first_log = logs[first.clamp_min(0)][:, None]
    first_log = torch.where(has_child, first_log, own+1.)
    retained = torch.where(has_child, 1.-ramp(coordinate, own, first_log),
                           torch.ones_like(coordinate))
    incoming_parts = []
    weights = []
    for level, sl in enumerate(layer_slices):
        n = sl.stop-sl.start
        if level == 0:
            incoming = torch.ones((n, 1), device=psi_current.device, dtype=psi_current.dtype)
        else:
            p = parents[sl]
            has_parent = (p >= 0)[:, None]
            safe_parent = p.clamp_min(0)
            parent_coordinate = coordinate[safe_parent]
            knot = logs[level]
            lower = logs[previous[sl].clamp_min(0)][:, None]
            lower = torch.where(has_parent, lower, knot-1.)
            next_level = following[sl]
            has_next = (next_level >= 0)[:, None]
            upper = logs[next_level.clamp_min(0)][:, None]
            upper = torch.where(has_next, upper, knot+1.)
            rising = ramp(parent_coordinate, lower, knot)
            falling = torch.where(has_next, 1.-ramp(parent_coordinate, knot, upper),
                                  torch.ones_like(rising))
            cohort = torch.minimum(rising, falling)
            earlier = torch.cat(incoming_parts, dim=0)
            incoming = torch.where(has_parent, earlier[safe_parent]*cohort,
                                   torch.ones_like(cohort))
        incoming_parts.append(incoming)
        weights.append(incoming*retained[sl])
    return torch.cat(weights, dim=0)


def render(model, camera, lod=True, max_level=None, require_depth=True, kernel_size=0.1, compact=True):
    """Render all included layers in one globally depth-sorted rasterization.

    lod=False disables only LoD modulation, not the per-layer sampling filter.
    f=fx in pixel units. Model tensors supply RaDe-GS filtered scales and
    determinant-compensated opacity; screen-space filtering remains in CUDA.
    """
    data = model.tensors(max_level=max_level)
    xyz = data['xyz']
    distance = (xyz - camera.center).norm(dim=1, keepdim=True)
    if not lod:
        weights = torch.ones_like(data['opacity'])
    elif model.topology_mode == 'legacy_flat':
        # Explicit diagnostic interpretation for ancestry-free old checkpoints.
        weights = lod_weight(distance / camera.fx, data['psi_ref'], model.step_scale)
    elif model.weight_policy == 'legacy_tent':
        weights = topology_weight(distance / camera.fx, data['psi_ref'], model.step_scale,
                                  data['has_parent'], data['has_child'])
    else:
        model.require_stage_records()
        weights = interval_weights(distance / camera.fx, data['psi_ref'],
                                   [stage['scale'] for stage in model.stage_records],
                                   data, data['layer_slices'])
    effective_opacity = data['opacity'] * weights
    screen = torch.zeros_like(xyz, requires_grad=True)
    if torch.is_grad_enabled():
        screen.retain_grad()
    # A zero-weight primitive contributes neither radiance nor derivatives
    # inside the flat part of Eq. (9), but CUDA still allocates its tile bins.
    # Compact before rasterizing; index_select scatters screen gradients back.
    indices = torch.nonzero(weights[:, 0] > 0, as_tuple=False)[:, 0] if lod and compact else None
    if indices is not None and indices.numel() == 0:
        zero = (screen.sum() + xyz.sum() + effective_opacity.sum() + data['sh'].sum()) * 0
        scalar = torch.zeros((1, camera.height, camera.width), device=xyz.device) + zero
        image = scalar.expand(3, -1, -1).contiguous()
        return dict(image=image, depth=scalar, median_depth=scalar, normal=image,
                    alpha=scalar, radii=torch.zeros(len(xyz), device=xyz.device, dtype=torch.int32),
                    means2d=screen, layer_slices=data['layer_slices'], weights=weights)
    def selected(tensor):
        return tensor if indices is None else tensor.index_select(0, indices)
    projection = torch.zeros((4, 4), device=xyz.device, dtype=xyz.dtype)
    near, far = 0.01, 1e4
    projection[0, 0] = 2 * camera.fx / camera.width
    projection[1, 1] = 2 * camera.fy / camera.height
    # CUDA ndc2Pix(v,S)=((v+1)*S-1)/2. Respect COLMAP pixel centers.
    projection[0, 2] = (2 * camera.cx + 1) / camera.width - 1
    projection[1, 2] = (2 * camera.cy + 1) / camera.height - 1
    projection[2, 2] = far / (far - near)
    projection[2, 3] = -far * near / (far - near)
    projection[3, 2] = 1
    settings = GaussianRasterizationSettings(
        image_height=camera.height, image_width=camera.width,
        tanfovx=camera.width/(2*camera.fx), tanfovy=camera.height/(2*camera.fy),
        kernel_size=kernel_size, bg=torch.zeros(3, device=xyz.device),
        scale_modifier=1., viewmatrix=camera.w2c.T.contiguous(),
        projmatrix=(projection @ camera.w2c).T.contiguous(),
        sh_degree=model.degree, campos=camera.center, prefiltered=False,
        require_depth=require_depth, debug=False,
    )
    image, radii, depth, median, alpha, normal = GaussianRasterizer(settings)(
        means3D=selected(xyz), means2D=selected(screen), opacities=selected(effective_opacity),
        shs=selected(data['sh']), scales=selected(data['scales']), rotations=selected(data['rotations']),
    )
    if indices is not None:
        radii = torch.zeros(len(xyz), device=xyz.device, dtype=radii.dtype).scatter_(0, indices, radii)
    return dict(image=image, depth=depth, median_depth=median, normal=normal,
                alpha=alpha, radii=radii, means2d=screen,
                layer_slices=data['layer_slices'], weights=weights)

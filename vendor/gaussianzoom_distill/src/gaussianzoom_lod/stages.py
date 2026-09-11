"""Zoom-stage protocol records for the GaussianZoom LoD component.

A stage record is plain, JSON/``torch.load(..., weights_only=True)``-safe data::

    {'scale': 2.0, 'cameras': [{'name': str, 'width': int, 'height': int,
                                'fx': float, 'fy': float, 'cx': float, 'cy': float,
                                'w2c': [[float] * 4] * 4}]}

``scale`` is always the measured ``median(fx / base.fx)`` of the stage's
cameras against the base stage -- never a nominal target and never inferred
from renderer primitives.  Tensor/numpy inputs are detached into plain data;
the poses are the rig-constant world-to-camera transforms.  Pure stdlib.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping

__all__ = ["capture_stage", "validate_next_stage", "validate_stage_records", "validate_stage_use"]

_ATOL = 1e-6        # absolute floor for near-zero element-wise comparisons
_SCALE_RTOL = 1e-6  # stored scale vs measured median fx ratio: an identity, not an estimate
_SINGULAR_DET = 1e-12


def _plain(value):
    """Detach torch tensors / numpy arrays into plain python scalars and lists."""
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        value = value.tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (list, tuple)):  # numpy
        value = value.tolist()
    return value


def _field(camera, key, where):
    if isinstance(camera, Mapping):
        if key not in camera:
            raise ValueError(f"{where}: camera missing field {key!r}")
        return camera[key]
    try:
        return getattr(camera, key)
    except AttributeError:
        raise ValueError(f"{where}: camera of type {type(camera).__name__} has no field {key!r}") from None


def _scalar(value, key, where):
    value = _plain(value)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{where}: field {key!r} must be a scalar, got sequence of length {len(value)}")
        value = _plain(value[0])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: field {key!r} must be a real number, got {type(value).__name__}")
    return float(value)


def _int_dim(value, key, where):
    value = _plain(value)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{where}: field {key!r} must be a scalar dimension, got sequence of length {len(value)}")
        value = _plain(value[0])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: field {key!r} must be an integer, got {type(value).__name__}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{where}: field {key!r} must be an integer, got {value!r}")
        value = int(value)
    if value <= 0:
        raise ValueError(f"{where}: field {key!r} must be positive, got {value}")
    return value


def _finite(value, what, where):
    if not math.isfinite(value):
        raise ValueError(f"{where}: {what} must be finite, got {value!r}")
    return value


def _det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _det4(m):
    return sum(sign * m[0][j] * _det3([[m[i][k] for k in range(4) if k != j] for i in range(1, 4)])
               for j, sign in zip(range(4), (1.0, -1.0, 1.0, -1.0)))


def _w2c(value, where):
    value = _plain(value)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{where}: w2c must be a 4x4 matrix, got {type(value).__name__}")
    rows = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            shape = f"row of length {len(row)}" if isinstance(row, (list, tuple)) else type(row).__name__
            raise ValueError(f"{where}: w2c must be a 4x4 matrix, got {shape}")
        rows.append([_finite(_scalar(cell, "w2c", where), "w2c entry", where) for cell in row])
    det = _det4(rows)
    if abs(det) <= _SINGULAR_DET:
        raise ValueError(f"{where}: w2c is singular (determinant {det:.3g})")
    return rows


def _parse_camera(camera, where):
    name = _field(camera, "name", where)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}: camera 'name' must be a non-empty string, got {name!r}")
    c = f"{where} camera {name!r}"
    fx = _finite(_scalar(_field(camera, "fx", where), "fx", c), "fx", c)
    fy = _finite(_scalar(_field(camera, "fy", where), "fy", c), "fy", c)
    parsed = {"name": name,
              "width": _int_dim(_field(camera, "width", where), "width", c),
              "height": _int_dim(_field(camera, "height", where), "height", c),
              "fx": fx, "fy": fy,
              "cx": _finite(_scalar(_field(camera, "cx", where), "cx", c), "cx", c),
              "cy": _finite(_scalar(_field(camera, "cy", where), "cy", c), "cy", c),
              "w2c": _w2c(_field(camera, "w2c", where), c)}
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"{c}: fx/fy must be positive, got fx={fx!r} fy={fy!r}")
    return parsed


def _parse_cameras(cameras, where):
    if cameras is None or isinstance(cameras, (str, bytes, Mapping)):
        raise ValueError(f"{where}: cameras must be a non-empty sequence, got {type(cameras).__name__}")
    try:
        items = list(cameras)
    except TypeError:
        raise ValueError(f"{where}: cameras must be a non-empty sequence, got {type(cameras).__name__}") from None
    if not items:
        raise ValueError(f"{where}: cameras must be a non-empty sequence")
    parsed, index = [], {}
    for i, camera in enumerate(items):
        cam = _parse_camera(camera, f"{where}[{i}]")
        if cam["name"] in index:
            raise ValueError(f"{where}: duplicate camera name {cam['name']!r}")
        index[cam["name"]] = cam
        parsed.append(cam)
    return parsed, index


def _same_pose(got, want, rtol, where):
    # Pixel-grid rounding may need 2% focal tolerance, not 2% pose motion.
    rtol = min(rtol, 1e-5)
    for r, (grow, wrow) in enumerate(zip(got, want)):
        for c, (g, w) in enumerate(zip(grow, wrow)):
            if not math.isclose(g, w, rel_tol=rtol, abs_tol=_ATOL):
                raise ValueError(f"{where}: w2c[{r}][{c}] moved away from the base stage: got {g!r}, base {w!r} (rtol={rtol:g})")


def _record(record, where):
    """Validate record structure; return (scale, cameras, index by name)."""
    if not isinstance(record, Mapping):
        raise ValueError(f"{where}: stage record must be a dict, got {type(record).__name__}")
    missing = [key for key in ("scale", "cameras") if key not in record]
    if missing:
        raise ValueError(f"{where}: stage record missing field(s) {missing}")
    scale = _finite(_scalar(record["scale"], "scale", where), "scale", where)
    if scale <= 0.0:
        raise ValueError(f"{where}: stage scale must be positive, got {scale!r}")
    cams, index = _parse_cameras(record["cameras"], f"{where} cameras")
    return scale, cams, index


def _require_base(record, where):
    scale, cams, index = _record(record, where)
    if scale != 1.0:
        raise ValueError(f"{where}: base stage scale must be exactly 1.0, got {scale!r}")
    return scale, cams, index


def _median_fx_ratio(base_index, stage_cams, where):
    ratios = []
    for cam in stage_cams:
        base = base_index.get(cam["name"])
        if base is None:
            raise ValueError(f"{where}: camera {cam['name']!r} is not present in the base stage")
        ratios.append(cam["fx"] / base["fx"])
    return statistics.median(ratios)


def _check_stored_scale(base_index, scale, stage_cams, where):
    measured = _median_fx_ratio(base_index, stage_cams, where)
    if not math.isclose(scale, measured, rel_tol=_SCALE_RTOL, abs_tol=0.0):
        raise ValueError(f"{where}: stored scale {scale!r} does not match the measured median fx ratio {measured!r} against the base stage")


def _checked_stage(base_index, stage_cams, prev_scale, step_scale, rtol, where):
    """Subset-of-base names, rig-constant poses, per-camera fx/fy step; return measured median fx ratio."""
    target = prev_scale * step_scale
    ratios = []
    for cam in stage_cams:
        name = cam["name"]
        base = base_index.get(name)
        if base is None:
            raise ValueError(f"{where}: camera {name!r} is not present in the base stage")
        c = f"{where} camera {name!r}"
        _same_pose(cam["w2c"], base["w2c"], rtol, c)
        for key in ("fx", "fy"):
            ratio = cam[key] / base[key]
            if not math.isclose(ratio, target, rel_tol=rtol, abs_tol=0.0):
                raise ValueError(f"{c}: {key} ratio {ratio:.6g} vs base does not match the expected step {target:.6g} (previous scale {prev_scale:.6g} * step_scale {step_scale:.6g}, rtol={rtol:g})")
            if key == "fx":
                ratios.append(ratio)
    return statistics.median(ratios)


def _rtol(rtol):
    value = _finite(_scalar(rtol, "rtol", "stage validation"), "rtol", "stage validation")
    if value < 0.0:
        raise ValueError(f"stage validation: rtol must be non-negative, got {value!r}")
    return value


def _step_scale(step_scale):
    value = _finite(_scalar(step_scale, "step_scale", "stage validation"), "step_scale", "stage validation")
    if value <= 1.0:
        raise ValueError(f"stage validation: step_scale must be > 1 to advance the focal scale (equal/decreased stages are rejected), got {value!r}")
    return value


def capture_stage(cameras, scale=1.0):
    """Snapshot ``cameras`` (dicts or attribute objects, tensors ok) into a plain stage record."""
    scale = _finite(_scalar(scale, "scale", "capture_stage"), "scale", "capture_stage")
    if scale <= 0.0:
        raise ValueError(f"capture_stage: scale must be positive, got {scale!r}")
    parsed, _ = _parse_cameras(cameras, "capture_stage")
    return {"scale": scale, "cameras": parsed}


def validate_next_stage(base_record, previous_record, cameras, step_scale, rtol=0.02):
    """Validate ``cameras`` as the stage following ``previous_record``; return its measured record."""
    rtol = _rtol(rtol)
    step_scale = _step_scale(step_scale)
    _, _, base_index = _require_base(base_record, "validate_next_stage: base_record")
    prev_scale, prev_cams, _ = _record(previous_record, "validate_next_stage: previous_record")
    _check_stored_scale(base_index, prev_scale, prev_cams, "validate_next_stage: previous_record")
    cams, _ = _parse_cameras(cameras, "validate_next_stage: cameras")
    measured = _checked_stage(base_index, cams, prev_scale, step_scale, rtol, "validate_next_stage")
    if not measured > prev_scale:
        raise ValueError(f"validate_next_stage: new stage scale {measured!r} does not increase on previous stage scale {prev_scale!r}")
    return {"scale": measured, "cameras": cams}


def validate_stage_use(record, cameras, rtol=1e-5):
    """Bind/densify-time check: ``cameras`` must be exactly the declared full active set of ``record``."""
    rtol = _rtol(rtol)
    _, _, index = _record(record, "validate_stage_use: record")
    cams, cam_index = _parse_cameras(cameras, "validate_stage_use: cameras")
    missing = sorted(set(index) - set(cam_index))
    extra = sorted(set(cam_index) - set(index))
    if missing or extra:
        raise ValueError(f"validate_stage_use: camera set does not match the declared stage (missing: {missing}, unexpected: {extra})")
    for cam in cams:
        want = index[cam["name"]]
        c = f"validate_stage_use camera {cam['name']!r}"
        if cam["width"] != want["width"] or cam["height"] != want["height"]:
            raise ValueError(f"{c}: raster {cam['width']}x{cam['height']} does not match the declared stage {want['width']}x{want['height']}")
        for key in ("fx", "fy", "cx", "cy"):
            if not math.isclose(cam[key], want[key], rel_tol=rtol, abs_tol=_ATOL):
                raise ValueError(f"{c}: {key} {cam[key]!r} does not match the declared stage {want[key]!r} (rtol={rtol:g})")
        _same_pose(cam["w2c"], want["w2c"], rtol, c)


def validate_stage_records(records, step_scale, rtol=0.02):
    """Validate the chain [base, stage1, ...]: subsets, rigid poses, focal steps, monotonic measured scales."""
    rtol = _rtol(rtol)
    step_scale = _step_scale(step_scale)
    if records is None or isinstance(records, (str, bytes, Mapping)):
        raise ValueError(f"validate_stage_records: records must be a non-empty sequence, got {type(records).__name__}")
    try:
        items = list(records)
    except TypeError:
        raise ValueError(f"validate_stage_records: records must be a non-empty sequence, got {type(records).__name__}") from None
    if not items:
        raise ValueError("validate_stage_records: records must be a non-empty sequence")
    _, _, base_index = _require_base(items[0], "validate_stage_records: record 0")
    prev_scale = 1.0
    for i, record in enumerate(items[1:], 1):
        where = f"validate_stage_records: record {i}"
        scale, cams, _ = _record(record, where)
        _checked_stage(base_index, cams, prev_scale, step_scale, rtol, where)
        if not scale > prev_scale:
            raise ValueError(f"{where}: stage scale {scale!r} does not increase on previous stage scale {prev_scale!r}")
        _check_stored_scale(base_index, scale, cams, where)
        prev_scale = scale

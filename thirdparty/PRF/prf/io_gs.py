from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from prf.types import GaussianScene

SH_C0 = 0.28209479177387814


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def quaternion_wxyz_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    quaternions = quaternions / np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-12)
    w, x, y, z = quaternions.T
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3)


def gaussian_min_axis_normals(rotations: np.ndarray, scales: np.ndarray) -> np.ndarray:
    matrices = quaternion_wxyz_to_matrix(rotations)
    min_axis = np.argmin(scales, axis=1)
    normals = matrices[np.arange(len(matrices)), :, min_axis]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return normals


def covariance_world(rotations: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """World-space covariance Sigma = R diag(s^2) R^T."""
    matrices = quaternion_wxyz_to_matrix(rotations)
    scaled = matrices * scales[:, None, :]
    return np.matmul(scaled, scaled.transpose(0, 2, 1))


def load_gaussians_ply(path: Path | str) -> GaussianScene:
    """Load a standard 3DGS PLY. scale_* are log-scale, opacity is logit."""
    path = Path(path)
    with path.open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} is missing end_header")
            header.append(line)
            if line.strip() == b"end_header":
                break
        payload = handle.read()

    props: list[str] = []
    count = None
    for raw in header:
        text = raw.decode("latin1").strip()
        if text.startswith("element vertex"):
            count = int(text.split()[-1])
        elif text.startswith("property float"):
            props.append(text.split()[-1])
    if count is None:
        raise ValueError(f"{path} is missing element vertex")
    expected = count * len(props) * 4
    if len(payload) != expected:
        raise ValueError(f"{path} payload size {len(payload)} != {expected}")

    data = np.frombuffer(payload, dtype="<f4").reshape(count, len(props))
    index = {name: i for i, name in enumerate(props)}
    required = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"{path} missing PLY properties: {missing}")

    mu = data[:, [index["x"], index["y"], index["z"]]].astype(np.float64)
    scales = np.exp(np.clip(data[:, [index["scale_0"], index["scale_1"], index["scale_2"]]], -20.0, 20.0)).astype(
        np.float64
    )
    rotations = data[:, [index["rot_0"], index["rot_1"], index["rot_2"], index["rot_3"]]].astype(np.float64)
    opacity = _sigmoid(data[:, index["opacity"]]).astype(np.float64)
    normals = gaussian_min_axis_normals(rotations, scales)
    rgb = None
    if all(f"f_dc_{axis}" in index for axis in range(3)):
        sh_dc = data[:, [index["f_dc_0"], index["f_dc_1"], index["f_dc_2"]]].astype(np.float64)
        rgb = np.clip(0.5 + SH_C0 * sh_dc, 0.0, 1.0)
    return GaussianScene(mu=mu, scales=scales, rotations=rotations, opacity=opacity, normals=normals, rgb=rgb)


def apply_similarity(scene: GaussianScene, scale: float, rotation: np.ndarray, translation: np.ndarray) -> GaussianScene:
    """Metricization: X' = s R X + t, Sigma' = s^2 R Sigma R^T."""
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    mu = scale * (scene.mu @ rotation.T) + translation
    matrices = quaternion_wxyz_to_matrix(scene.rotations)
    new_R = rotation @ matrices
    # Convert rotation matrices back to wxyz quaternions.
    rotations = _matrices_to_quaternions(new_R)
    return GaussianScene(
        mu=mu,
        scales=scene.scales * scale,
        rotations=rotations,
        opacity=scene.opacity.copy(),
        normals=None if scene.normals is None else scene.normals @ rotation.T,
        rgb=None if scene.rgb is None else scene.rgb.copy(),
    )


def _matrices_to_quaternions(matrices: np.ndarray) -> np.ndarray:
    # Shepperd's method, batched.
    m00, m01, m02 = matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 0, 2]
    m10, m11, m12 = matrices[:, 1, 0], matrices[:, 1, 1], matrices[:, 1, 2]
    m20, m21, m22 = matrices[:, 2, 0], matrices[:, 2, 1], matrices[:, 2, 2]
    trace = m00 + m11 + m22
    quats = np.empty((matrices.shape[0], 4), dtype=np.float64)
    t0 = trace > 0
    s0 = np.sqrt(np.maximum(trace[t0] + 1.0, 0.0)) * 2.0
    quats[t0, 0] = 0.25 * s0
    quats[t0, 1] = (m21[t0] - m12[t0]) / s0
    quats[t0, 2] = (m02[t0] - m20[t0]) / s0
    quats[t0, 3] = (m10[t0] - m01[t0]) / s0

    rest = ~t0
    c0 = rest & (m00 >= m11) & (m00 >= m22)
    s = np.sqrt(np.maximum(1.0 + m00[c0] - m11[c0] - m22[c0], 0.0)) * 2.0
    quats[c0, 0] = (m21[c0] - m12[c0]) / s
    quats[c0, 1] = 0.25 * s
    quats[c0, 2] = (m01[c0] + m10[c0]) / s
    quats[c0, 3] = (m02[c0] + m20[c0]) / s

    c1 = rest & ~c0 & (m11 >= m22)
    s = np.sqrt(np.maximum(1.0 + m11[c1] - m00[c1] - m22[c1], 0.0)) * 2.0
    quats[c1, 0] = (m02[c1] - m20[c1]) / s
    quats[c1, 1] = (m01[c1] + m10[c1]) / s
    quats[c1, 2] = 0.25 * s
    quats[c1, 3] = (m12[c1] + m21[c1]) / s

    c2 = rest & ~c0 & ~c1
    s = np.sqrt(np.maximum(1.0 + m22[c2] - m00[c2] - m11[c2], 0.0)) * 2.0
    quats[c2, 0] = (m10[c2] - m01[c2]) / s
    quats[c2, 1] = (m02[c2] + m20[c2]) / s
    quats[c2, 2] = (m12[c2] + m21[c2]) / s
    quats[c2, 3] = 0.25 * s
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    return quats


def assert_metric(scene: GaussianScene, cfg) -> None:
    if not scene.is_metric(cfg.metric_extent_min_m, cfg.metric_extent_max_m, cfg.metric_scale_min_m, cfg.metric_scale_max_m):
        extent = float(np.median(np.ptp(scene.mu, axis=0)))
        scale = float(np.median(scene.scales.max(axis=1)))
        raise ValueError(
            "Gaussian scene does not look metric-scale: "
            f"median extent={extent:.4g} m, median max-axis scale={scale:.4g} m. "
            "Apply metricization before computing R_phys."
        )


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        return math.inf
    weights = np.maximum(weights, 0.0)
    if float(weights.sum()) <= 0.0:
        return float(np.quantile(values, q))
    order = np.argsort(values)
    values = values[order]
    cdf = np.cumsum(weights[order])
    cdf = cdf / cdf[-1]
    return float(np.interp(q, cdf, values))

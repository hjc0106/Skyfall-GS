"""Explicit similarities and the exact Stage 1 Camera projection convention."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from utils.graphics_utils import getProjectionMatrix, getWorld2View2, focal2fov


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def similarity(scale, rotation, translation):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Similarity scale must be finite and positive")
    rotation = np.asarray(rotation, dtype=np.float64)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0:
        raise ValueError("Expected a proper rotation")
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix


def transform_camera(c2w, matrix):
    """Change world coordinates by a similarity without scaling camera rotation."""
    matrix = np.asarray(matrix, dtype=np.float64)
    scale = np.cbrt(np.linalg.det(matrix[:3, :3]))
    if scale <= 0:
        raise ValueError("Invalid camera similarity")
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3] @ np.asarray(c2w)[:3, :3] / scale
    result[:3, 3] = transform_points(np.asarray(c2w)[None, :3, 3], matrix)[0]
    return result


def current_reader_transform(contents, xyz):
    """Match readSatelliteInfo; missing R/T means identity, not an inferred ENU transform."""
    if 'R' not in contents:
        return np.eye(4)
    rotation = np.asarray(contents['R'], dtype=np.float64)[:3, :3]
    translation = np.asarray(contents['T'], dtype=np.float64)
    aligned = xyz @ rotation.T - translation
    scale = 256.0 / np.percentile(np.linalg.norm(aligned, axis=1), 99)
    z_shift = np.percentile((aligned * scale)[:, 2], 1)
    return similarity(scale, rotation, -translation * scale - [0, 0, z_shift])


def camera_matrices(frame, width, height, resolution=1):
    """Use precisely Camera's float32 view/projection construction."""
    c2w = np.asarray(frame.get('transform_matrix_rotated', frame['transform_matrix']), dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    view = torch.tensor(getWorld2View2(w2c[:3, :3].T, w2c[:3, 3])).T
    projection = getProjectionMatrix(
        0.01, 100.0, focal2fov(frame['fl_x'], width), focal2fov(frame['fl_y'], height),
        2 * frame['cx'] / width - 1, 2 * frame['cy'] / height - 1,
    ).T
    return view, view @ projection, (round(width / resolution), round(height / resolution))


def make_training_camera(frame, width, height, resolution=4):
    """Instantiate the repository's Camera without importing scene's training dependencies."""
    path = Path(__file__).resolve().parents[1] / 'scene/cameras.py'
    spec = importlib.util.spec_from_file_location('_stage1_actual_camera', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c2w = np.asarray(frame.get('transform_matrix_rotated', frame['transform_matrix']))
    w2c = np.linalg.inv(c2w)
    return module.Camera(
        colmap_id=0, uid=0, R=w2c[:3, :3].T, T=w2c[:3, 3],
        FoVx=focal2fov(frame['fl_x'], width), FoVy=focal2fov(frame['fl_y'], height),
        cx=2 * frame['cx'] / width - 1, cy=2 * frame['cy'] / height - 1,
        image=torch.zeros(3, round(height / resolution), round(width / resolution)),
        gt_alpha_mask=None, image_name=frame['file_path'], data_device='cpu',
    )


def project_matrices(points, view, full_projection, size):
    device = full_projection.device
    hom = torch.tensor(np.c_[points, np.ones(len(points))], dtype=torch.float32, device=device)
    camera = hom @ view
    clip = hom @ full_projection
    ndc = clip[:, :2] / (clip[:, 3:4] + 1e-7)
    # cuda_rasterizer/auxiliary.h::ndc2Pix, including its half-pixel convention.
    pixels = ((ndc + 1) * torch.tensor(size, device=device) - 1) * 0.5
    return pixels.detach().cpu().numpy().astype(np.float64), camera[:, 2].detach().cpu().numpy()


def project_frame(points, frame, width, height, resolution=4, device='cpu'):
    if device == 'cuda':
        camera = make_training_camera(frame, width, height, resolution)
        return project_matrices(points, camera.world_view_transform, camera.full_proj_transform,
                                (camera.image_width, camera.image_height))
    view, full, size = camera_matrices(frame, width, height, resolution)
    return project_matrices(points, view, full, size)


def estimate_similarity(points, observation_points, frame_indices, frames, initial=None):
    """Fit one world similarity to verified training track observations (float64 optimization).

    observation_points use the pinhole pixel convention, before ndc2Pix's -0.5.
    Require broad multiview support; acceptance is a separate independent gate.
    """
    if len(points) < 30 or len(np.unique(frame_indices)) < 3:
        raise ValueError('Need at least 30 observations in at least three training cameras')
    points = np.asarray(points)
    initial = np.eye(4) if initial is None else np.asarray(initial)
    center = np.median(points, axis=0)
    radius = max(np.percentile(np.linalg.norm(points - center, axis=1), 90), 1.0)
    normalized = (points - center) / radius
    scale0 = np.cbrt(np.linalg.det(initial[:3, :3]))
    rotation0 = initial[:3, :3] / scale0
    initial_center = transform_points(center[None], initial)[0]
    origins = np.array([np.asarray(f['transform_matrix'])[:3, 3] for f in frames])
    camera_scale = max(np.median(np.linalg.norm(origins - initial_center, axis=1)), 1.0)
    w2cs = np.array([np.linalg.inv(f['transform_matrix']) for f in frames])[frame_indices]
    intrinsic = np.array([[f['fl_x'], f['fl_y'], f['cx'], f['cy']] for f in frames])[frame_indices]

    def unpack(params):
        rotation = Rotation.from_rotvec(params[:3]).as_matrix()
        scale = np.exp(params[3])
        translation = initial_center + params[4:] * camera_scale
        return scale, rotation, translation

    def residual(params):
        scale, rotation, translation = unpack(params)
        world = normalized @ rotation.T * scale + translation
        cam = np.einsum('nij,nj->ni', w2cs[:, :3, :3], world) + w2cs[:, :3, 3]
        z = cam[:, 2:3]
        z = np.where(np.abs(z) < 1e-8, 1e-8, z)
        xy = cam[:, :2] / z * intrinsic[:, :2] + intrinsic[:, 2:]
        return (xy - observation_points).ravel()

    params = np.r_[Rotation.from_matrix(rotation0).as_rotvec(), np.log(scale0 * radius), [0, 0, 0]]
    fit = least_squares(residual, params, loss='huber', f_scale=2.0, max_nfev=500, x_scale='jac')
    scale, rotation, translation = unpack(fit.x)
    matrix = similarity(scale / radius, rotation, translation - scale / radius * (rotation @ center))
    return matrix, {'success': bool(fit.success), 'nfev': fit.nfev, 'cost': float(fit.cost)}

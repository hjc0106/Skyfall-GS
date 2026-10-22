#!/usr/bin/env python3
"""Stage 1 coordinate preflight. A failed preflight must not launch training.

This first phase diagnoses source data only; it does not manufacture missing
2D observations or emit an unvalidated DSM-to-training transform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from priors.coordinates import current_reader_transform, project_frame, transform_points
from priors.dsm import read_dsm, sample_dsm


def read_tracks(path):
    ids, xyz, rgb, errors, tracks = [], [], [], [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 8 or (len(parts) - 8) % 2:
            raise ValueError('Malformed COLMAP point/track record')
        ids.append(int(parts[0]))
        xyz.append(list(map(float, parts[1:4])))
        rgb.append(list(map(int, parts[4:7])))
        errors.append(float(parts[7]))
        tracks.append(np.asarray(parts[8:], dtype=np.int64).reshape(-1, 2))
    xyz = np.asarray(xyz)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Point cloud must contain finite points')
    return np.asarray(ids), xyz, np.asarray(rgb), np.asarray(errors), tracks


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {k: float(v) for k, v in zip(('min', 'median', 'p95', 'max'),
                                        np.percentile(values, [0, 50, 95, 100]))}


def inventory_rpc(directory, frames, tracks):
    """Describe other observations, never assume their indices belong to COLMAP tracks."""
    if directory is None:
        return None
    directory = Path(directory)
    train_names = {Path(f['file_path']).stem for f in frames}
    rows = []
    for path in sorted(directory.glob('*.json')):
        if path.stem not in train_names:
            continue
        data = json.loads(path.read_text())
        kp = data.get('keypoints', {})
        xy = kp.get('2d_coordinates', [])
        indices = kp.get('pts3d_indices', [])
        rows.append({'path': str(path.resolve()), 'image': data.get('img'),
                     'width': data.get('width'), 'height': data.get('height'),
                     'observation_count': len(xy),
                     'pts3d_index_range': [int(min(indices)), int(max(indices))] if indices else None})
    points_path = directory / 'pts3d.npy'
    cloud = np.load(points_path) if points_path.exists() else None
    return {
        'directory': str(directory.resolve()), 'training_name_matches': len(rows), 'images': rows,
        'associated_cloud': str(points_path.resolve()),
        'associated_point_count': len(cloud) if cloud is not None else None,
        'associated_median_xyz': np.median(cloud, axis=0).tolist() if cloud is not None else None,
        'colmap_feature_index_max': max(int(t[:, 1].max()) for t in tracks if len(t)),
        'usable_as_current_sfm_tracks': False,
        'reason': 'RPC crop observations index a separate pts3d.npy; image-name overlap does not establish COLMAP feature identity or crop-to-training pixel mapping.',
    }


def load_observations(path, point_ids, xyz, tracks, frames):
    """Read genuine COLMAP images.txt/bin, checking both sides of each track link."""
    if path is None:
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location('_stage1_colmap_io', ROOT / 'scene/colmap_loader.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = Path(path)
    images = (module.read_extrinsics_binary if path.suffix == '.bin' else module.read_extrinsics_text)(str(path))
    by_name = {Path(f['file_path']).stem: i for i, f in enumerate(frames)}
    rows = []
    for point_id, point, track in zip(point_ids, xyz, tracks):
        for image_id, feature_index in track:
            image = images.get(int(image_id))
            if image is None or Path(image.name).stem not in by_name:
                continue
            if feature_index < 0 or feature_index >= len(image.xys):
                raise ValueError(f'Feature index {feature_index} is absent in image {image_id}')
            if image.point3D_ids[feature_index] != point_id:
                raise ValueError(f'Point ID mismatch for image {image_id}, feature {feature_index}')
            rows.append((point, image.xys[feature_index], by_name[Path(image.name).stem]))
    if not rows:
        raise ValueError('No verified observations match the training split')
    points, pixels, frame_ids = zip(*rows)
    return np.asarray(points), np.asarray(pixels), np.asarray(frame_ids)


def diagnose(args):
    source = args.source.resolve()
    point_path = source / 'points3D.txt'
    transform_path = source / 'transforms_train.json'
    contents = json.loads(transform_path.read_text())
    frames = contents['frames']
    width, height = contents['w'], contents['h']
    point_ids, xyz, _, stored_errors, tracks = read_tracks(point_path)
    reader_matrix = current_reader_transform(contents, xyz)
    current_xyz = transform_points(xyz, reader_matrix)
    if 'R' in contents:
        # Rotated camera poses in source already include R/T; reader only scales
        # their translations and applies its final z shift.
        scale = np.cbrt(np.linalg.det(reader_matrix[:3, :3]))
        aligned = xyz @ np.asarray(contents['R'])[:3, :3].T - np.asarray(contents['T'])
        z_shift = np.percentile((aligned * scale)[:, 2], 1)
        frames = [dict(f) for f in frames]
        for frame in frames:
            pose = np.asarray(frame['transform_matrix_rotated']).copy()
            pose[:3, 3] = pose[:3, 3] * scale - [0, 0, z_shift]
            frame['transform_matrix_rotated'] = pose.tolist()
    report = {
        'schema_version': 1, 'base_commit': '22134e125fe9c4ee21d3bfcc105ec89275b26dcc',
        'phase': 'source_coordinate_preflight', 'ready_for_training': False,
        'training_started': False, 'resolution': args.resolution, 'projection_backend': args.device,
        'source': str(source), 'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in (point_path, transform_path)},
        'current_reader': {'has_R': 'R' in contents, 'has_T': 'T' in contents,
                           'sfm_to_current_training': reader_matrix.tolist(),
                           'current_training_to_sfm': np.linalg.inv(reader_matrix).tolist()},
        'sfm': {'count': len(xyz), 'median_xyz': np.median(xyz, axis=0).tolist(),
                'radius99_from_origin_m': float(np.percentile(np.linalg.norm(xyz, axis=1), 99)),
                'z_m': stats(xyz[:, 2]), 'track_observations': sum(len(t) for t in tracks),
                'stored_COLMAP_error_not_current_reprojection': stats(stored_errors)},
        'current_reprojection_error_px': None, 'current_projection': [], 'blockers': [],
    }
    center = np.median(current_xyz, axis=0)
    for frame in frames:
        pixels, depth = project_frame(current_xyz, frame, width, height, args.resolution, args.device)
        size = (round(width / args.resolution), round(height / args.resolution))
        finite = np.isfinite(pixels).all(axis=1) & np.isfinite(depth)
        in_image = finite & (depth > .2) & (pixels >= 0).all(axis=1) & (pixels < size).all(axis=1)
        c2w = np.asarray(frame.get('transform_matrix_rotated', frame['transform_matrix']))
        report['current_projection'].append({
            'image': frame['file_path'], 'camera_to_sfm_center_m': float(np.linalg.norm(c2w[:3, 3] - center)),
            'positive_depth_points': int((depth > 0).sum()), 'in_image_and_rasterizer_depth_points': int(in_image.sum()),
            'depth': stats(depth), 'pixel_x': stats(pixels[:, 0]), 'pixel_y': stats(pixels[:, 1]),
            'within_Camera_near_far_points': int(((depth >= .01) & (depth <= 100)).sum()),
        })
    observations = load_observations(args.observations, point_ids, current_xyz, tracks, frames)
    if observations is None:
        report['blockers'].append('missing_verified_training_2d_observations')
    else:
        points, measured, frame_ids = observations
        errors = []
        for i, frame in enumerate(frames):
            mask = frame_ids == i
            if mask.any():
                pixels, _ = project_frame(points[mask], frame, width, height, args.resolution, args.device)
                # Compare in the actual rasterizer pixel convention.
                errors.extend(np.linalg.norm(pixels - (measured[mask] / args.resolution - .5), axis=1))
        report['current_reprojection_error_px'] = stats(errors)
        if np.median(errors) > args.max_reprojection_error:
            report['blockers'].append('current_camera_sfm_reprojection_failed_needs_similarity_fit')
    if not any(r['in_image_and_rasterizer_depth_points'] for r in report['current_projection']):
        report['blockers'].append('no_sfm_points_project_into_any_training_image')
    report['rpc_crop_observations'] = inventory_rpc(args.rpc_observations, frames, tracks)
    heights, valid, dsm_info = read_dsm(args.dsm, args.dsm_sidecar, args.dsm_epsg, args.dsm_nodata)
    samples = sample_dsm(heights, valid, dsm_info, args.spacing_m, args.max_dsm_points)
    report['dsm'] = {**dsm_info, 'sample_count': len(samples), 'spacing_m': args.spacing_m,
                     'sample_bounds_map': [samples.min(axis=0).tolist(), samples.max(axis=0).tolist()]}
    report['blockers'].append('dsm_to_camera_and_final_normalization_not_validated')
    report['coordinate_roundtrip'] = {'current_reader_max_abs_error': float(np.abs(
        transform_points(current_xyz, np.linalg.inv(reader_matrix)) - xyz).max()),
        'final_registered_training_transform': None}
    report['accepted'] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dsm', type=Path, required=True)
    parser.add_argument('--dsm-sidecar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observations', type=Path, help='Verified COLMAP images.txt or images.bin')
    parser.add_argument('--rpc-observations', type=Path, help='Inventory RPC crop JSONs; never use their indices as COLMAP tracks')
    parser.add_argument('--dsm-epsg', type=int, choices=[32617], default=32617)
    parser.add_argument('--dsm-nodata', type=float)
    parser.add_argument('--resolution', type=int, default=4)
    parser.add_argument('--spacing-m', type=float, default=2.0)
    parser.add_argument('--max-dsm-points', type=int, default=20000)
    parser.add_argument('--max-reprojection-error', type=float, default=2.0, help='Pixels at requested resolution')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if args.resolution <= 0:
        parser.error('resolution must be positive')
    output = args.output.resolve()
    # This command only writes to its explicit output. Reject output inside input dataset.
    if output == args.source.resolve() or args.source.resolve() in output.parents:
        parser.error('output must be outside the source dataset')
    output.mkdir(parents=True, exist_ok=True)
    try:
        report = diagnose(args)
    except Exception as error:
        report = {'accepted': False, 'ready_for_training': False, 'training_started': False,
                  'blockers': [f'{type(error).__name__}: {error}']}
    (output / 'coordinate_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    smoke = {'accepted': False, 'training_started': False, 'phase': 'coordinate_gate',
             'coordinate_gate_passed': False, 'abc': {k: {'status': 'not_started'} for k in 'ABC'},
             'coordinate_report': 'coordinate_report.json', 'blockers': report['blockers'],
             'note': 'No training or reconstruction quality claim; stop at a failed coordinate gate.'}
    (output / 'smoke_report.json').write_text(json.dumps(smoke, indent=2) + '\n')
    print(json.dumps({'accepted': False, 'report': str(output / 'coordinate_report.json'),
                      'blockers': report['blockers']}, indent=2))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())

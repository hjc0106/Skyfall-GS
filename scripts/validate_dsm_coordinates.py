#!/usr/bin/env python3
"""Validate an existing B/C prior against genuine training-track observations.

This validates the source B/C metric frame; it never estimates a transform from
unknown feature indices, and never changes a registered prior or source data.
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
from priors.coordinates import project_frame, transform_points
from scripts.prepare_dsm_prior import read_tracks, load_observations, stats
from utils.dsm_prior import load_npz


def validate(args):
    source = args.source.resolve()
    contents = json.loads((source / 'transforms_train.json').read_text())
    if 'R' in contents:
        raise ValueError('Source B/C metric-frame validator does not accept R/T-normalized datasets')
    ids, xyz, _, _, tracks = read_tracks(source / 'points3D.txt')
    frames = contents['frames']
    observations = load_observations(args.observations, ids, xyz, tracks, frames)
    if observations is None:
        raise ValueError('Verified COLMAP images.txt/bin observations are required')
    points, measured, indices = observations
    if len(points) < 30 or len(np.unique(indices)) < 3:
        raise ValueError('Need at least 30 verified observations in three training views')
    prior = load_npz(args.prior)
    registration_path = args.prior.with_name('registration.json')
    registration = json.loads(registration_path.read_text())
    transform = registration['transform']
    if transform.get('E') != 'E0 + x' or transform.get('N') != 'N0 + y' or transform.get('H') != 'h0 + z + dz':
        raise ValueError('Unsupported registration: expected source B/C metric-frame translation')
    world_to_map = np.eye(4)
    world_to_map[:3, 3] = [transform['E0'], transform['N0'], transform['h0'] + transform['dz']]
    recovered = transform_points(transform_points(xyz, world_to_map), np.linalg.inv(world_to_map))
    roundtrip = float(np.abs(recovered - xyz).max())
    rows, cols = np.where(prior['valid'])
    select = np.linspace(0, len(rows) - 1, min(20000, len(rows)), dtype=int)
    rows, cols = rows[select], cols[select]
    dsm_xyz = np.c_[prior['x0'] + cols * prior['gsd'], prior['y0'] - rows * prior['gsd'], prior['height_world'][rows, cols]]
    supported = np.zeros(len(rows), bool)
    errors, positive = [], []
    for i, frame in enumerate(frames):
        mask = indices == i
        if mask.any():
            pixels, depth = project_frame(points[mask], frame, contents['w'], contents['h'], args.resolution, args.device)
            errors.extend(np.linalg.norm(pixels - (measured[mask] / args.resolution - .5), axis=1))
            positive.extend(depth > .2)
        pixels, depth = project_frame(dsm_xyz, frame, contents['w'], contents['h'], args.resolution, args.device)
        size = np.array([round(contents['w'] / args.resolution), round(contents['h'] / args.resolution)])
        supported |= np.isfinite(pixels).all(axis=1) & (depth > .2) & (pixels >= 0).all(axis=1) & (pixels < size).all(axis=1)
    errors = np.asarray(errors)
    checks = {'roundtrip': roundtrip < 1e-8, 'finite_reprojection': bool(np.isfinite(errors).all()),
              'median_reprojection': bool(np.median(errors) <= args.max_error),
              'p95_reprojection': bool(np.percentile(errors, 95) <= args.max_error * 3),
              'positive_track_depth': bool(np.all(positive)), 'dsm_support': bool(supported.all())}
    return {'accepted': all(checks.values()), 'ready_for_training': all(checks.values()),
            'current_reprojection_error_px': stats(errors), 'checks': checks,
            'blockers': [key for key, value in checks.items() if not value],
            'source_hashes': {str(source / name): hashlib.sha256((source / name).read_bytes()).hexdigest()
                              for name in ('points3D.txt', 'transforms_train.json')},
            'prior_sha256': hashlib.sha256(args.prior.read_bytes()).hexdigest(),
            'observations_sha256': hashlib.sha256(args.observations.read_bytes()).hexdigest(),
            'coordinate_roundtrip_max_abs': roundtrip, 'dsm_supported_fraction': float(supported.mean()),
            'training_world_to_map': world_to_map.tolist(), 'map_to_training_world': np.linalg.inv(world_to_map).tolist(),
            'normalization_scale': 1.0, 'normalization_source': 'preserve existing source B/C metric frame',
            'resolution': args.resolution, 'projection_backend': args.device}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=4)
    parser.add_argument('--max-error', type=float, default=2.)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if args.resolution <= 0 or args.max_error <= 0:
        parser.error('resolution and max-error must be positive')
    try:
        report = validate(args)
    except Exception as error:
        report = {'accepted': False, 'ready_for_training': False, 'current_reprojection_error_px': None,
                  'blockers': [f'{type(error).__name__}: {error}']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Four-step synthetic C smoke: pause/resume versus uninterrupted training.

This uses known camera/geometry coordinates and does not validate JAX_068 or
measure reconstruction quality. It never touches a real input dataset.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--port', type=int, default=6298)
    args = parser.parse_args()
    output = args.output.resolve()
    source = output / 'source'
    (source / 'images').mkdir(parents=True, exist_ok=True)
    row, col = np.mgrid[:64, :64]
    rgb = np.stack([40 + 2 * col + row // 3, 50 + col // 3 + 2 * row, 70 + col + row], axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(source / 'images/test.png')
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1., -1., -1.])
    pose[:3, 3] = [3., 3., 10.]
    frame = dict(file_path='./images/test.png', fl_x=60., fl_y=60., cx=32., cy=32., transform_matrix=pose.tolist())
    for split in ('train', 'test'):
        (source / f'transforms_{split}.json').write_text(json.dumps(dict(w=64, h=64, frames=[frame])))
    points = [[x, y, .05 * x + .1 * y] for x in np.linspace(1, 5, 6) for y in np.linspace(1, 5, 6)]
    (source / 'points3D.txt').write_text(''.join(f'{i} {x} {y} {z} 100 100 100 0\n' for i, (x, y, z) in enumerate(points)))
    normal = np.broadcast_to(np.array([-.05, -.1, 1.], np.float32), (8, 8, 3)).copy()
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    rows, cols = np.mgrid[:8, :8]
    height = (.05 * cols + .1 * (7 - rows)).astype(np.float32)
    prior = output / 'prior.npz'
    np.savez(prior, height_world=height, normal=normal,
             confidence=np.ones((8, 8), np.float32), valid=np.ones((8, 8), bool),
             holdout=np.zeros((8, 8), bool), edges=np.array([[0, 1]], np.int64),
             x0=0., y0=7., gsd=1., sample_step=2, coarse_h=4, coarse_w=4)
    common = [sys.executable, str(ROOT / 'train.py'), '-s', str(source), '--eval', '--resolution', '1',
              '--sh_degree', '1', '--appearance_enabled', '--lambda_depth', '0', '--lambda_pseudo_depth', '0',
              '--dsm_prior', str(prior), '--iterations', '4', '--test_iterations', '4',
              '--checkpoint_iterations', '2', '4', '--save_iterations', '2', '4', '--densify_from_iter', '1',
              '--densification_interval', '1', '--densify_until_iter', '4', '--densify_grad_threshold', '0.00000001',
              '--opacity_reset_interval', '2', '--size_threshold', '100000', '--dsm_warmup_iters', '1',
              '--verify_finite_gradients', '--port', str(args.port)]
    resumed, continuous = output / 'resumed', output / 'continuous'
    commands = [common + ['-m', str(resumed), '--stop_after_iteration', '2'],
                common + ['-m', str(resumed), '--start_checkpoint', str(resumed / 'chkpnt2.pth')],
                common + ['-m', str(continuous)]]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4')
    report = {'accepted': False, 'kind': 'synthetic_only', 'iterations': 4,
              'commands': commands, 'reconstruction_quality_claim': False}
    try:
        for index, command in enumerate(commands):
            with (output / f'phase_{index}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        # Compare CPU copies; this parent need not allocate model memory on a GPU.
        a = torch.load(resumed / 'chkpnt4.pth', map_location='cpu', weights_only=False)[0]
        b = torch.load(continuous / 'chkpnt4.pth', map_location='cpu', weights_only=False)[0]
        differences = {}
        for index in (1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 15, 16):
            x, y = a['model_state'][index], b['model_state'][index]
            if x is None:
                assert y is None
                continue
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
            differences[str(index)] = float((x - y).detach().abs().max()) if x.numel() else 0.
        # Tangent-isotropic splats have a redundant rotation degree of freedom.
        # CUDA reduction noise can change raw quaternions even before the pause.
        # Compare their geometric covariance, and retain raw differences in the report.
        from scipy.spatial.transform import Rotation
        def covariance(payload):
            state = payload['model_state']
            quaternion = state[5].detach().numpy()
            rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()
            variance = np.exp(2 * state[4].detach().numpy())
            return (rotation * variance[:, None, :]) @ rotation.transpose(0, 2, 1)
        cov_a, cov_b = covariance(a), covariance(b)
        np.testing.assert_allclose(cov_a, cov_b, rtol=1e-5, atol=1e-6)
        report['covariance_max_abs_difference'] = float(np.abs(cov_a - cov_b).max())
        report['rotation_parameter_max_abs_difference'] = float((a['model_state'][5] - b['model_state'][5]).detach().abs().max())
        report['rotation_comparison'] = 'geometric_covariance; raw quaternion is redundant for tangent-isotropic splats'
        assert torch.equal(a['rng']['torch'], b['rng']['torch']) and a['rng']['python'] == b['rng']['python']
        for key, tensor in a['model_state'][9].state_dict().items():
            torch.testing.assert_close(tensor, b['model_state'][9].state_dict()[key], rtol=1e-5, atol=1e-6)
        events = json.loads((resumed / 'stage1_events.json').read_text())
        assert events['resumed'] and events['events']['finite_gradient_steps'] == 4 and events['prior_alignment']
        assert events['events']['cloned'] > 0 and events['events']['opacity_resets'] > 0
        with (output / 'evaluate.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_stage1.py'), '--model', str(resumed),
                            '--source', str(source), '--prior', str(prior), '--iteration', '4'],
                           cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        report.update(accepted=True, events=events, resume_max_abs_differences=differences)
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'synthetic_smoke_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('accepted', 'kind', 'events', 'resume_max_abs_differences')}, indent=2))


if __name__ == '__main__':
    main()

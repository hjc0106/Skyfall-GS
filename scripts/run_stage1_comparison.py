#!/usr/bin/env python3
"""Portable A/B/C commands, with a strict coordinate gate before training."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_coordinate_gate(report_path, source, prior):
    if report_path is None:
        raise ValueError('A validated --coordinate-report is required before training')
    report = json.loads(Path(report_path).read_text())
    if not report.get('accepted') or not report.get('ready_for_training') or report.get('current_reprojection_error_px') is None:
        raise ValueError('Coordinate gate failed or measured training-track reprojection is unverified')
    expected = report.get('source_hashes', {})
    for name in ('points3D.txt', 'transforms_train.json'):
        values = [v for k, v in expected.items() if Path(k).name == name]
        actual = hashlib.sha256((Path(source) / name).read_bytes()).hexdigest()
        if values != [actual]:
            raise ValueError('Coordinate report source hash does not match ' + name)
    if report.get('prior_sha256') != hashlib.sha256(Path(prior).read_bytes()).hexdigest():
        raise ValueError('Coordinate gate was not validated for this registered prior')
    return report


def commands(args):
    profile = json.loads((ROOT / 'configs' / (args.profile + '.json')).read_text())
    steps = 200 if args.smoke else (args.iterations or profile['iterations'])
    result = {}
    for group in ('A', 'B', 'C'):
        output = args.output.resolve() / f'{group}_s{args.seed}'
        command = [sys.executable, str(ROOT / 'train.py'), '-s', str(args.source.resolve()),
                   '-m', str(output), '--seed', str(args.seed), '--dsm_seed', str(args.seed),
                   '--port', str(args.port), *profile['arguments'], '--iterations', str(steps),
                   '--checkpoint_iterations', str(steps), '--save_iterations', str(steps),
                   '--test_iterations', str(steps)]
        if group != 'A':
            command += ['--dsm_prior', str(args.prior.resolve())]
        if group == 'B':
            command += ['--no_dsm_surface']
        if args.smoke:
            command += ['--resolution', '4', '--lambda_pseudo_depth', '0', '--lambda_depth', '0',
                        '--init_max_points', str(args.max_init_points), '--dsm_warmup_iters', '20',
                        '--densify_from_iter', '50', '--densification_interval', '50',
                        '--densify_until_iter', '200', '--opacity_reset_interval', '100',
                        '--checkpoint_iterations', '100', '200', '--save_iterations', '100', '200',
                        '--verify_finite_gradients']
            first = command + ['--stop_after_iteration', '100']
            second = command + ['--start_checkpoint', str(output / 'chkpnt100.pth')]
            result[group] = [first, second]
        else:
            result[group] = [command]
        result[group].append([sys.executable, str(ROOT / 'scripts/evaluate_stage1.py'),
                              '--model', str(output), '--source', str(args.source.resolve()),
                              '--prior', str(args.prior.resolve()), '--iteration', str(steps)])
    return result, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coordinate-report', type=Path)
    parser.add_argument('--profile', choices=['stage1_dsm', 'source_bc_replay'], default='stage1_dsm')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--max-init-points', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--port', type=int, default=6218)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan, steps = commands(args)
    gate_error = None
    try:
        check_coordinate_gate(args.coordinate_report, args.source, args.prior)
    except (ValueError, OSError) as error:
        gate_error = str(error)
    report = {'accepted': False, 'training_started': False, 'coordinate_gate_passed': gate_error is None,
              'blockers': [gate_error] if gate_error else [], 'profile': args.profile,
              'commands': plan, 'abc': {key: {'status': 'not_started'} for key in plan}}
    path = args.output / 'smoke_report.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    if gate_error:
        parser.error(gate_error)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4')
    try:
        for group, group_commands in plan.items():
            report['training_started'] = True
            path.write_text(json.dumps(report, indent=2) + '\n')
            output = args.output.resolve() / f'{group}_s{args.seed}'
            output.mkdir(parents=True, exist_ok=True)
            with (output / 'run.log').open('w') as log:
                for command in group_commands:
                    subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            evidence = json.loads((output / 'stage1_events.json').read_text())
            if args.smoke:
                events = evidence['events']
                if not evidence['resumed'] or not evidence['prior_alignment']:
                    raise ValueError('Resume or prior alignment was not verified')
                if events['finite_loss_steps'] != steps or events['finite_gradient_steps'] != steps:
                    raise ValueError('Finite loss/gradient checks did not cover every step')
                if min(events['densification_calls'], events['opacity_resets']) < 1:
                    raise ValueError('Smoke did not exercise densification and reset')
                if events['cloned'] + events['split_added'] <= 0 or events['pruned'] <= 0:
                    raise ValueError('Smoke did not actually add and prune Gaussian points')
                if group == 'C' and evidence['surface_active_points'] <= 0:
                    raise ValueError('C surface loss had no active points')
            report['abc'][group].update(status='completed', checks=evidence)
        from scripts.evaluate_stage1 import common_grid_reports
        from utils.dsm_prior import load_npz
        comparison = common_grid_reports({g: args.output / f'{g}_s{args.seed}/dsm_pred_iter_{steps}.npz' for g in plan}, load_npz(args.prior))
        (args.output / 'common_grid_metrics.json').write_text(json.dumps(comparison, indent=2) + '\n')
        report['accepted'] = True
    except Exception as error:
        report['blockers'].append(str(error))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

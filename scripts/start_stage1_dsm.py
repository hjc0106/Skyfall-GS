#!/usr/bin/env python3
"""Find original COLMAP observations, enforce the coordinate gate, then run A/B/C."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observations', type=Path, help='Original matching COLMAP images.txt/bin')
    parser.add_argument('--search-root', type=Path, action='append', default=[],
                        help='Directory to search recursively; repeat for multiple original SfM output directories')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--full', action='store_true', help='Run full profile instead of the default 200-step smoke')
    parser.add_argument('--validate-only', action='store_true', help='Validate candidates without starting training')
    parser.add_argument('--profile', choices=['stage1_dsm', 'source_bc_replay'], default='stage1_dsm')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.source = args.source.expanduser().resolve()
    args.prior = args.prior.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'startup_report.json'
    report = {'ready_for_training': False, 'training_started': False, 'blockers': [],
              'candidates': [], 'source': str(args.source), 'prior': str(args.prior)}
    required = [args.source / name for name in ('images', 'points3D.txt', 'transforms_train.json')]
    required += [args.prior, args.prior.with_name('registration.json')]
    report['blockers'] = ['Missing required input: ' + str(p) for p in required if not p.exists()]
    roots = [args.source, *[p.expanduser().resolve() for p in args.search_root]]
    candidates = set()
    if args.observations:
        observation = args.observations.expanduser().resolve()
        if not observation.is_file():
            report['blockers'].append('Explicit observations file does not exist: ' + str(observation))
        else:
            candidates.add(observation)
    else:
        for root in roots:
            if not root.is_dir():
                report['blockers'].append('Search directory does not exist: ' + str(root))
                continue
            for directory, _, names in os.walk(root):
                for name in ('images.bin', 'images.txt'):
                    if name in names:
                        candidates.add((Path(directory) / name).resolve())
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    accepted = None
    if not report['blockers']:
        for index, candidate in enumerate(sorted(candidates)):
            gate_path = args.output / f'coordinate_candidate_{index}.json'
            command = [sys.executable, str(ROOT / 'scripts/validate_dsm_coordinates.py'),
                       '--source', str(args.source), '--prior', str(args.prior),
                       '--observations', str(candidate), '--resolution', '4', '--output', str(gate_path)]
            print('Validating original observations: ' + str(candidate), flush=True)
            with (args.output / f'coordinate_candidate_{index}.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
            item = {'observations': str(candidate), 'exit_code': result.returncode,
                    'report': str(gate_path), 'accepted': bool(gate.get('accepted')),
                    'blockers': gate.get('blockers', ['Validator did not write a report'])}
            report['candidates'].append(item)
            if result.returncode == 0 and gate.get('accepted') and gate.get('ready_for_training'):
                accepted = gate_path
                break
        if accepted is None:
            report['blockers'].append('No matching observations passed the coordinate gate.' if candidates else
                'No original COLMAP images.txt/bin found. Retrieve the SfM output that generated points3D.txt; '
                'the public Skyfall-GS and SatNeRF packages do not supply verified matching observations. '
                'Re-running COLMAP creates a different reconstruction and cannot fill this gap for the existing tracks.')
    report['ready_for_training'] = accepted is not None and not report['blockers']
    if report['ready_for_training']:
        report['coordinate_report'] = str(accepted)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    if not report['ready_for_training']:
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 2
    if args.validate_only:
        print('Coordinate gate passed. Report: ' + str(report_path))
        return 0
    command = [sys.executable, str(ROOT / 'scripts/run_stage1_comparison.py'),
               '--source', str(args.source), '--prior', str(args.prior), '--output', str(args.output),
               '--coordinate-report', str(accepted), '--profile', args.profile, '--gpu', args.gpu,
               '--seed', str(args.seed)]
    if not args.full:
        command.append('--smoke')
    report['training_started'] = True
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return subprocess.run(command, cwd=ROOT, env=environment).returncode


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download prepared Skyfall-GS scenes from the upstream Hugging Face dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='JAX_068', help='Scene such as JAX_068 or NYC_004')
    parser.add_argument('--output', type=Path, default=Path('data'), help='Parent of datasets_JAX / datasets_NYC')
    parser.add_argument('--revision', default='main', help='Upstream revision; resolved to a commit before download')
    parser.add_argument('--dry-run', action='store_true', help='List remote scene files without downloading payloads')
    args = parser.parse_args()
    if not re.fullmatch(r'(JAX|NYC)_\d{3}', args.scene):
        parser.error('--scene must have the form JAX_068 or NYC_004')
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        parser.error('Install huggingface-hub (the repository pins 0.33.4) before running this script')
    repo = 'jayinnn/Skyfall-GS-datasets'
    prefix = f'datasets_{args.scene[:3]}/{args.scene}/'
    api = HfApi()
    revision = api.repo_info(repo_id=repo, repo_type='dataset', revision=args.revision).sha
    files = sorted(name for name in api.list_repo_files(repo_id=repo, repo_type='dataset', revision=revision)
                   if name.startswith(prefix))
    if not files:
        parser.error(f'No files found for {args.scene} at revision {revision}')
    print(f'Repository: {repo}\nRevision: {revision}\nScene files: {len(files)}')
    if args.dry_run:
        print('\n'.join(files))
        return
    destination = args.output.expanduser().resolve()
    snapshot_download(repo_id=repo, repo_type='dataset', revision=revision,
                      allow_patterns=[prefix + '*'], local_dir=str(destination))
    scene = destination / prefix
    missing = [name for name in ('images', 'transforms_train.json', 'points3D.txt')
               if not (scene / name).exists()]
    if missing:
        parser.error('Downloaded scene is missing required inputs: ' + ', '.join(missing))
    print(f'Dataset: {scene}')
    print('DSM TIFF/TXT, scene.loc and matching COLMAP observations must be prepared separately.')


if __name__ == '__main__':
    main()

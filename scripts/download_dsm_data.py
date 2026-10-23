#!/usr/bin/env python3
"""Fetch scene DSMs from a SatNeRF release, the DFC2019 torrent, or a local ZIP."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

TORRENT_URL = ('https://raw.githubusercontent.com/pubgeo/dfc2019/master/data/'
               'DFC2019_track3_trainval_v1.0.0.torrent')

SATNERF_URL = ('https://github.com/centreborelli/satnerf/releases/download/'
               'EarthVision2022/dataset.zip')


def decode_torrent(data):
    stream = io.BytesIO(data)
    def parse():
        token = stream.read(1)
        if token == b'i':
            return int(until(b'e'))
        if token in (b'l', b'd'):
            items = []
            while True:
                pos = stream.tell()
                if stream.read(1) == b'e':
                    break
                stream.seek(pos)
                items.append(parse())
            return dict(zip(items[::2], items[1::2])) if token == b'd' else items
        if token.isdigit():
            return stream.read(int(token + until(b':')))
        raise ValueError('Invalid torrent metadata')
    def until(delimiter):
        result = b''
        while True:
            char = stream.read(1)
            if not char:
                raise ValueError('Truncated torrent')
            if char == delimiter:
                return result
            result += char
    return parse()


def fetch(url, path):
    temporary = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Skyfall-GS-data-downloader'})
            with urllib.request.urlopen(request, timeout=30) as response, temporary.open('wb') as out:
                shutil.copyfileobj(response, out)
            temporary.replace(path)
            return
        except (OSError, ValueError):
            temporary.unlink(missing_ok=True)
            print(f'HTTPS download attempt {attempt + 1}/3 failed.', file=sys.stderr)
            if attempt < 2:
                time.sleep(attempt + 1)
    # Some server networks work with curl HTTP/1.1 but interrupt Python TLS.
    if shutil.which('curl'):
        result = subprocess.run(['curl', '--http1.1', '--fail', '--location',
                                 '--retry', '2', '--connect-timeout', '15', '--max-time', '3600',
                                 '--output', str(temporary), url])
        if result.returncode == 0:
            temporary.replace(path)
            return
        temporary.unlink(missing_ok=True)
    raise RuntimeError('HTTPS download failed after retries. Check network/proxy, or use '
                       '--torrent-file for existing metadata / --archive for an existing truth ZIP.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='JAX_068')
    parser.add_argument('--output', type=Path, default=Path('data/DFC2019/Track3-Truth'))
    parser.add_argument('--source', choices=['satnerf', 'torrent'], default='satnerf',
                        help='Default: author-published SatNeRF HTTPS ZIP; torrent is optional')
    parser.add_argument('--cache', type=Path, default=Path('downloads/dfc2019'))
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--archive', type=Path, help='Import a previously downloaded truth ZIP')
    source.add_argument('--torrent-file', type=Path, help='Use local torrent metadata without fetching GitHub')
    source.add_argument('--url', help='Download a truth ZIP from an actual accessible file URL')
    parser.add_argument('--list-only', action='store_true', help='List scene files (SatNeRF needs its cached/downloaded ZIP; torrent needs metadata only)')
    args = parser.parse_args()
    if not re.fullmatch(r'(JAX|OMA)_\d{3}', args.scene):
        parser.error('--scene must have the form JAX_068 or OMA_001')
    args.cache.mkdir(parents=True, exist_ok=True)
    files = []
    if args.archive:
        files = [args.archive]
    elif args.url:
        archive = args.cache / 'truth.zip'
        fetch(args.url, archive)
        files = [archive]
    elif args.source == 'satnerf' and not args.torrent_file:
        archive = args.cache / 'satnerf_EarthVision2022_dataset.zip'
        if not archive.is_file():
            print('Downloading author-published SatNeRF dataset (~640 MB): ' + SATNERF_URL, flush=True)
            fetch(SATNERF_URL, archive)
        else:
            print('Using cached SatNeRF dataset: ' + str(archive))
        if not zipfile.is_zipfile(archive):
            parser.error('Cached SatNeRF ZIP is invalid; remove or replace ' + str(archive))
        files = [archive]
        if args.list_only:
            with zipfile.ZipFile(archive) as z:
                print('\n'.join(n for n in z.namelist() if Path(n).name in
                                {args.scene + '_DSM.tif', args.scene + '_DSM.txt'}))
            return
    else:
        torrent = args.torrent_file or args.cache / 'track3_trainval.torrent'
        if args.torrent_file and not torrent.is_file():
            parser.error('Local torrent file does not exist: ' + str(torrent))
        if not torrent.is_file():
            fetch(TORRENT_URL, torrent)
        else:
            print('Using existing torrent metadata: ' + str(torrent))
        try:
            info = decode_torrent(torrent.read_bytes())[b'info']
        except (ValueError, KeyError, TypeError):
            parser.error('Invalid torrent metadata: ' + str(torrent) +
                         '; replace it with a valid official torrent before retrying')
        root = Path(info[b'name'].decode())
        entries = info.get(b'files')
        if entries is None:
            parser.error('Expected a multi-file Track 3 torrent')
        selected = []
        for index, entry in enumerate(entries, 1):
            relative = Path(*(part.decode() for part in entry[b'path']))
            if relative.is_absolute() or '..' in relative.parts or root.is_absolute() or '..' in root.parts:
                parser.error('Unexpected unsafe torrent path')
            name = str(relative).lower()
            if 'truth' in name and ('track3' in name or 'track_3' in name):
                selected.append(index)
                files.append(args.cache / root / relative)
                print(f'{index}: {relative} ({entry[b"length"] / 1024**2:.1f} MiB)')
        if not selected:
            parser.error('No Track 3 truth files found; inspect the official data page')
        if args.list_only:
            return
        if not shutil.which('aria2c'):
            parser.error('Install aria2 first: conda install -c conda-forge aria2')
        subprocess.run(['aria2c', '--seed-time=0', '--continue=true', '--bt-remove-unselected-file=true',
                        '--select-file=' + ','.join(map(str, selected)),
                        '--dir=' + str(args.cache.resolve()), str(torrent.resolve())], check=True)
    wanted = {args.scene + '_DSM.tif', args.scene + '_DSM.txt'}
    args.output.mkdir(parents=True, exist_ok=True)
    found = {}
    for archive in files:
        if archive.name in wanted and archive.is_file():
            found[archive.name] = archive.read_bytes()
        elif archive.is_file() and zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                for member in z.infolist():
                    name = Path(member.filename).name
                    if name in wanted:
                        if name in found:
                            raise ValueError('Ambiguous duplicate scene file: ' + name)
                        found[name] = z.read(member)
    if set(found) != wanted:
        parser.error('Scene DSM TIFF/TXT pair was not found in downloaded truth files')
    import numpy as np
    import tifffile
    image = tifffile.imread(io.BytesIO(found[args.scene + '_DSM.tif']))
    bounds = [float(v) for v in found[args.scene + '_DSM.txt'].decode().split()]
    if len(bounds) != 4 or bounds[2] <= 0 or bounds[3] <= 0 or image.shape != (int(bounds[2]), int(bounds[2])) or not np.isfinite(image).any():
        parser.error('DSM TIFF and sidecar validation failed')
    for name, data in found.items():
        target = args.output / name
        if target.exists() and target.read_bytes() != data:
            parser.error('Refusing to overwrite a different file: ' + str(target))
    for name, data in found.items():
        (args.output / name).write_bytes(data)
    report = {'scene': args.scene, 'source': 'local_archive' if args.archive else ('provided_url' if args.url else (SATNERF_URL if args.source == 'satnerf' and not args.torrent_file else TORRENT_URL)),
              'shape': list(image.shape), 'sidecar': bounds,
              'sha256': {name: hashlib.sha256(data).hexdigest() for name, data in found.items()}}
    (args.output / (args.scene + '_download.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'Download failed: {error}', file=sys.stderr)
        sys.exit(1)

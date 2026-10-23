#!/usr/bin/env python3
"""Fetch DFC2019 Track 3 truth via its official torrent, or import a truth ZIP."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request
import zipfile

TORRENT_URL = ('https://raw.githubusercontent.com/pubgeo/dfc2019/master/data/'
               'DFC2019_track3_trainval_v1.0.0.torrent')


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
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as out:
        shutil.copyfileobj(response, out)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='JAX_068')
    parser.add_argument('--output', type=Path, default=Path('data/DFC2019/Track3-Truth'))
    parser.add_argument('--cache', type=Path, default=Path('downloads/dfc2019'))
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--archive', type=Path, help='Import a previously downloaded truth ZIP')
    source.add_argument('--url', help='Download a truth ZIP from an actual accessible file URL')
    parser.add_argument('--list-only', action='store_true', help='List selected official torrent files without downloading payloads')
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
    else:
        torrent = args.cache / 'track3_trainval.torrent'
        fetch(TORRENT_URL, torrent)
        info = decode_torrent(torrent.read_bytes())[b'info']
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
    report = {'scene': args.scene, 'source': 'local_archive' if args.archive else ('provided_url' if args.url else TORRENT_URL),
              'shape': list(image.shape), 'sidecar': bounds,
              'sha256': {name: hashlib.sha256(data).hexdigest() for name, data in found.items()}}
    (args.output / (args.scene + '_download.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

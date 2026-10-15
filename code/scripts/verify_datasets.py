"""Verify every shipped dataset file against its byte count and SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    manifest = json.loads((args.root/'datasets/MANIFEST.json').read_text())
    for entry in manifest['files']:
        path = args.root/entry['path']
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                digest.update(chunk)
        if path.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
            raise ValueError(f'Dataset checksum mismatch: {entry["path"]}')
    print(f'OK: {len(manifest["files"])} dataset files verified.')


if __name__ == '__main__':
    main()

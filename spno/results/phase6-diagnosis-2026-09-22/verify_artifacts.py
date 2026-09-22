"""Verify the transferred checkpoints/data using only Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,required=True)
    args=parser.parse_args()
    root=args.repo_root.resolve()
    manifest=json.loads(Path(__file__).with_name('local-artifacts-manifest.json').read_text(encoding='utf-8'))
    failed=[]
    for item in manifest['artifacts']:
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root):
            failed.append(item['path']+': invalid path'); continue
        if not path.is_file():
            failed.append(item['path']+': missing'); continue
        if path.stat().st_size!=item['bytes']:
            failed.append(item['path']+': size mismatch'); continue
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(4*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=item['sha256']:
            failed.append(item['path']+': SHA-256 mismatch')
    if failed:
        print('\n'.join(failed))
        raise SystemExit(f'FAILED: {len(failed)} artifact(s). Do not resume evaluation yet.')
    print(f"PASS: {manifest['n_checkpoints']} checkpoints and {manifest['n_datasets']} data files match their recorded SHA-256 hashes.")

if __name__=='__main__':
    main()

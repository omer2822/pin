"""Build the source ZIP consumed by the five Colab notebooks (no datasets/weights)."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build_bundle(output: Path | None = None) -> Path:
    output = ROOT / 'dist' / 'spno-colab-source.zip' if output is None else Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / 'pyproject.toml', ROOT / 'README.md']
    sources.extend((ROOT / 'src').rglob('*.py'))
    sources.extend((ROOT / 'scripts').glob('*.py'))
    sources.extend((ROOT / 'notebooks').glob('*.ipynb'))
    sources.append(ROOT / 'notebooks' / 'README.md')
    temporary = output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sources):
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Source bundle failed its integrity check')
    temporary.replace(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    print(build_bundle(parser.parse_args().output))

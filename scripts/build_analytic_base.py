"""Deprecated compatibility shim for the NetMob derived-data build.

Canonical entrypoint: notebooks/00_build_preprocessed_datasets.ipynb
This script is kept only so existing manual CLI habits do not break abruptly.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / 'notebooks' / '00_build_preprocessed_datasets.ipynb'
    if not notebook_path.exists():
        raise FileNotFoundError(f'Notebook not found: {notebook_path}')

    print('DEPRECATED: use notebooks/00_build_preprocessed_datasets.ipynb as the canonical build artifact.')
    print('Running the notebook code path for backward-compatible CLI execution...')

    notebook = json.loads(notebook_path.read_text())
    namespace = {'__name__': '__main__'}
    for cell in notebook['cells']:
        if cell.get('cell_type') != 'code':
            continue
        source = ''.join(cell.get('source', []))
        exec(compile(source, f"{notebook_path.name}::{cell.get('id', 'code-cell')}", 'exec'), namespace)


if __name__ == '__main__':
    sys.exit(main())

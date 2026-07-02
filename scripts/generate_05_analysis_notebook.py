from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == '__main__':
    generator = Path(__file__).with_name('generate_analysis_notebooks.py')
    runpy.run_path(str(generator), run_name='__main__')
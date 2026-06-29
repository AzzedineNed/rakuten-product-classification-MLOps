"""Importing this module puts <project>/src on sys.path so entrypoints can do
`from rakuten_img import ...` without `pip install -e .` or setting PYTHONPATH.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

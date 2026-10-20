"""Shared by every tests/test_*.py so each file runs standalone with `python3 tests/test_x.py`
as well as under pytest from the repo root."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

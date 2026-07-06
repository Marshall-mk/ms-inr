"""Add the project root to sys.path so ``methods/<name>/run.py`` can ``import msinr``
without an editable install. Import this first in every method script.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

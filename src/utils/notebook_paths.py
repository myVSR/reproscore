"""
src/utils/notebook_paths.py
============================
Shared utility for finding Jupyter notebooks in a repository,
with consistent exclusion of non-project directories.

Excluded:
  .ipynb_checkpoints  — Jupyter auto-save
  .reproscore         — our own executed-notebook output
  site-packages       — installed Python packages (repos with embedded venvs)
  venv / .venv / env  — virtual environments
  node_modules        — JS dependencies
  __pycache__         — compiled Python bytecode
  .git                — version control internals
  lib/pythonX.Y       — embedded interpreter lib dirs (e.g. odecell/ENVODE/lib/python3.7)
"""

from __future__ import annotations

from pathlib import Path
from typing import Set

# Directory names (any component of the path) that indicate non-project content
_EXCLUDED_DIRS: Set[str] = {
    ".ipynb_checkpoints",
    ".reproscore",
    "site-packages",
    "venv",
    ".venv",
    "env",
    "node_modules",
    "__pycache__",
    ".git",
    ".tox",
    ".nox",
    "dist",
    "build",
    "eggs",
    ".eggs",
}

# Prefixes that indicate an embedded interpreter lib dir  e.g. "lib/python3.7"
_LIB_PYTHON_PREFIX = "lib"
_PYTHON_DIR_PREFIX = "python"


def _has_embedded_python_lib(parts: tuple) -> bool:
    """Return True if the path contains a lib/pythonX.Y segment."""
    for i, part in enumerate(parts):
        if part == _LIB_PYTHON_PREFIX and i + 1 < len(parts):
            nxt = parts[i + 1]
            if nxt.startswith(_PYTHON_DIR_PREFIX) and len(nxt) > len(_PYTHON_DIR_PREFIX):
                return True
    return False


def is_excluded_notebook(path: Path) -> bool:
    """
    Return True if a notebook path should be excluded from analysis.
    Checks all components of the path.
    """
    parts = path.parts
    for part in parts:
        if part in _EXCLUDED_DIRS:
            return True
    return _has_embedded_python_lib(parts)


def find_notebooks(repo: Path) -> list[Path]:
    """
    Return all .ipynb files in repo that are project notebooks
    (i.e. not in any excluded directory).
    Results are sorted by path for determinism.
    """
    return sorted(
        p for p in repo.rglob("*.ipynb")
        if not is_excluded_notebook(p)
    )

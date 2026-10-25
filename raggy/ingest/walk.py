"""Find the files to ingest under a folder, honouring ignore rules.

Hidden files and folders, node_modules and __pycache__ are always skipped.
Patterns in the folder's .raggyignore and .gitignore (gitignore syntax) are
skipped too. Ignored folders are never entered, so a 200,000-file
node_modules costs nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

from pathspec import GitIgnoreSpec

from raggy.ingest.loaders import SUFFIXES

ALWAYS_IGNORE = ["node_modules/", "__pycache__/"]
IGNORE_FILES = (".raggyignore", ".gitignore")


def _spec(root: Path) -> GitIgnoreSpec:
    lines = list(ALWAYS_IGNORE)
    for name in IGNORE_FILES:
        f = root / name
        if f.is_file():
            lines += f.read_text(encoding="utf-8", errors="replace").splitlines()
    return GitIgnoreSpec.from_lines(lines)


def iter_files(root: Path) -> list[Path]:
    root = Path(root)
    spec = _spec(root)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and not spec.match_file((rel / d).as_posix() + "/")]
        for name in filenames:
            if name.startswith(".") or Path(name).suffix.lower() not in SUFFIXES:
                continue
            if not spec.match_file((rel / name).as_posix()):
                found.append(Path(dirpath) / name)
    return sorted(found)

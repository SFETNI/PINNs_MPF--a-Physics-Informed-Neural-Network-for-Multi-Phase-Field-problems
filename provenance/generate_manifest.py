#!/usr/bin/env python
"""Deterministic provenance-manifest generator for this package.

Run from anywhere; it locates the package root as the parent of this file's
directory (this script lives in ``provenance/``) and writes:

  - provenance/MANIFEST.json         {package, generated_by, file_count, files:[{path,size_bytes,sha256}]}
  - provenance/PACKAGE_CONTENTS.tsv  path<TAB>size_bytes<TAB>sha256  (+ header)

Inclusion policy: hash every shipped file EXCEPT version-control / cache noise
(any ``.git/``, ``__pycache__/``, ``.pytest_cache/``, ``.ipynb_checkpoints/``
path segment and ``*.pyc`` / ``*.pyo`` / ``.DS_Store`` files) and the two manifest files
themselves (a manifest cannot hash itself). ``legacy_version/`` and ``media/``
are hashed in full.

Deterministic: files are sorted by POSIX relative path and sha256 is content
only, so the same tree in produces an identical manifest out.

    python provenance/generate_manifest.py

Verify afterwards, from the package root:

    awk -F'\\t' 'NR>1{print $3"  "$1}' provenance/PACKAGE_CONTENTS.tsv | sha256sum -c
"""
from __future__ import annotations

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))

EXCLUDE_SEGMENTS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo")
EXCLUDE_NAMES = {".DS_Store"}
EXCLUDE_RELPATHS = {"provenance/MANIFEST.json", "provenance/PACKAGE_CONTENTS.tsv"}

GENERATED_BY = ("provenance/generate_manifest.py — deterministic sha256 walk (excludes .git, "
                "__pycache__, .pytest_cache, *.pyc/.pyo, .DS_Store, and the manifest files "
                "themselves; hashes every other shipped file including legacy_version/ and media/)")


def sha256_of(path: str) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def included(relpath: str) -> bool:
    if relpath in EXCLUDE_RELPATHS:
        return False
    parts = relpath.split("/")
    if any(seg in EXCLUDE_SEGMENTS for seg in parts):
        return False
    name = parts[-1]
    if name in EXCLUDE_NAMES or name.endswith(EXCLUDE_SUFFIXES):
        return False
    return True


def main() -> int:
    entries = []
    for root, dirs, files in os.walk(PKG):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_SEGMENTS]
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, PKG).replace(os.sep, "/")
            if not included(rel):
                continue
            digest, size = sha256_of(full)
            entries.append({"path": rel, "size_bytes": size, "sha256": digest})

    entries.sort(key=lambda e: e["path"])

    manifest = {
        "package": "PINNS_MPF_offline",
        "generated_by": GENERATED_BY,
        "file_count": len(entries),
        "files": entries,
    }
    prov_dir = os.path.join(PKG, "provenance")
    os.makedirs(prov_dir, exist_ok=True)
    with open(os.path.join(prov_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    with open(os.path.join(prov_dir, "PACKAGE_CONTENTS.tsv"), "w") as f:
        f.write("path\tsize_bytes\tsha256\n")
        for e in entries:
            f.write(f"{e['path']}\t{e['size_bytes']}\t{e['sha256']}\n")

    print(f"manifest regenerated: {len(entries)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

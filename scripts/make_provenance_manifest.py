#!/usr/bin/env python
"""Regenerate provenance/MANIFEST.json and provenance/PACKAGE_CONTENTS.tsv.

Deterministic packaging-integrity generator: walks the package, hashes every file
(sorted by path with SHA-256), and writes the manifest + a flat TSV. Excludes the two
provenance outputs themselves and VCS/build cruft. No timestamps are recorded, so the
output is reproducible: same tree -> byte-identical manifest.

Run from anywhere:
    python scripts/make_provenance_manifest.py

Verify later, e.g.:
    python - <<'PY'
    import json,hashlib,os
    for e in json.load(open('provenance/MANIFEST.json'))['files']:
        h=hashlib.sha256(open(e['path'],'rb').read()).hexdigest()
        assert h==e['sha256'], e['path']
    print('manifest OK')
    PY
"""
from __future__ import annotations

import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # package root
PACKAGE = "PINNS_MPF_offline"
EXCLUDE_FILES = {"provenance/MANIFEST.json", "provenance/PACKAGE_CONTENTS.tsv"}
EXCLUDE_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> list[str]:
    rels = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn.endswith(".pyc"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), ROOT).replace(os.sep, "/")
            if rel in EXCLUDE_FILES:
                continue
            rels.append(rel)
    return sorted(rels)


def main() -> None:
    entries = [{"path": p,
                "size_bytes": os.path.getsize(os.path.join(ROOT, p)),
                "sha256": sha256(os.path.join(ROOT, p))}
               for p in collect()]
    manifest = {"package": PACKAGE,
                "generated_by": "scripts/make_provenance_manifest.py",
                "file_count": len(entries),
                "files": entries}
    prov = os.path.join(ROOT, "provenance")
    os.makedirs(prov, exist_ok=True)
    with open(os.path.join(prov, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    with open(os.path.join(prov, "PACKAGE_CONTENTS.tsv"), "w") as f:
        f.write("path\tsize_bytes\tsha256\n")
        for e in entries:
            f.write(f"{e['path']}\t{e['size_bytes']}\t{e['sha256']}\n")
    print(f"wrote provenance/MANIFEST.json + PACKAGE_CONTENTS.tsv: {len(entries)} files")


if __name__ == "__main__":
    main()

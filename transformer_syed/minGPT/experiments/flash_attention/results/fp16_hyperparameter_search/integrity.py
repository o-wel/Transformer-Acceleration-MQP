"""Refuse execution if any frozen source or dataset changed after submission."""
import hashlib
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
manifest=json.loads((root/'SNAPSHOT_SHA256.json').read_text())
for rel,expected in manifest.items():
    path=root/'snapshot/repo'/rel
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise RuntimeError(f'Frozen snapshot changed: {rel}')
print(f'Verified {len(manifest)} frozen source/data files',flush=True)

"""Check published documentation and provenance without running compute jobs."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
errors = []
files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts]
for p in files:
    rel = p.relative_to(ROOT).as_posix()
    if p.stat().st_size > 2 * 1024 * 1024:
        errors.append(f'Unexpected large file: {rel}')
    if p.suffix.lower() in {'.sif', '.safetensors', '.npz', '.tar', '.gz', '.zip', '.pem'}:
        errors.append(f'Unexpected binary/archive: {rel}')
    if p.suffix == '.md':
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            target = target.split('#')[0]
            if target and not (p.parent / target).exists():
                errors.append(f'Broken link in {rel}: {target}')
manifest = json.loads((ROOT/'docs/source-manifest.json').read_text(encoding='utf-8'))
for entry in manifest:
    path = ROOT / entry['path']
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        errors.append(f'Source provenance mismatch: {entry["path"]}')
with (ROOT/'benchmarks/rsm-progression.csv').open(encoding='utf-8') as f:
    for row in csv.DictReader(f):
        record = json.loads((ROOT/f'benchmarks/evidence/rsm-{row["version"]}.json').read_text())
        ratio = record['cases'][0]['metrics']['performance']['value']
        if float(row['public_speedup']) != ratio:
            errors.append(f'Benchmark mismatch: {row["version"]}')
if errors:
    raise SystemExit('\n'.join(errors))
print(f'PASS: {len(files)} files; documentation links, source hashes and RSM metrics checked.')
print('This is a repository check, not hardware correctness or performance validation.')

import hashlib
import json
from pathlib import Path
import random
from experiments.sports.data import TRAIN_TEAMS,example

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
STUDY=ROOT/'experiments/paper_hyperparameters/runs/study-20261105-v2'


def main():
    target=HERE/'fresh-quality.jsonl'
    if target.exists():raise RuntimeError('Refuse to overwrite frozen inputs')
    paths=list((ROOT/'experiments/sports').glob('*.jsonl'))
    paths+=list((STUDY/'snapshot/repo/experiments/paper_hyperparameters/data').glob('*.jsonl'))
    paths += [ROOT/f'experiments/{name}/quality-data.jsonl' for name in ('joint_decode','graph_decode')]
    seen=set()
    for p in paths:
        for line in p.read_text().splitlines():
            row=json.loads(line)
            if 'match' in row:
                m=row['match'];seen.add(tuple(m['half']+m['final']))
    rng=random.Random(20261110);rows=[]
    for i in range(120):
        while True:
            teams=rng.sample(TRAIN_TEAMS,2);half=rng.sample(range(25,56),2);final=rng.sample(range(65,106),2)
            sig=tuple(half+final)
            if sig not in seen:seen.add(sig);break
        m=dict(id=f'optimal-fresh-{i:03d}',teams=teams,half=half,final=final)
        rows.extend(example(m,kind) for kind in ('winner','halftime','comeback'))
    target.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    inherited=json.loads((STUDY/'selection.json').read_text())
    assert sha(STUDY/'selection.json')==(STUDY/'selection.sha256').read_text().strip()
    manifest=dict(data_sha256=sha(target),protocol_sha256=sha(HERE/'PROTOCOL.md'),seed=20261110,matches=120,questions=360,
        inherited_selection_sha256=sha(STUDY/'selection.json'),lora=inherited['lora'],cache_chunk=inherited['chunk'],
        excluded_sources={str(p.relative_to(ROOT)):sha(p) for p in paths})
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2));(HERE/'results').mkdir(exist_ok=True)
    print('Frozen 360 fresh questions and inherited hyperparameter provenance')


if __name__=='__main__':main()

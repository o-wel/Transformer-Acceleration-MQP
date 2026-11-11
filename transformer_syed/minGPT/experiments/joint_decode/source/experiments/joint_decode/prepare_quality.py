import hashlib
import json
from pathlib import Path
import random
from experiments.sports.data import TRAIN_TEAMS, example

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def main():
    out=HERE/'quality-data.jsonl'
    if out.exists():raise RuntimeError('Refuse overwrite')
    paths=list((ROOT/'experiments/sports').glob('*.jsonl'))
    paths+=list((ROOT/'experiments/paper_hyperparameters/runs/study-20261105-v2/snapshot/repo/experiments/paper_hyperparameters/data').glob('*.jsonl'))
    seen=set()
    for p in paths:
        for line in p.read_text().splitlines():
            r=json.loads(line)
            if 'match' in r:
                m=r['match'];seen.add(tuple(m['half']+m['final']))
    rng=random.Random(20261107);rows=[]
    for i in range(60):
        while True:
            teams=rng.sample(TRAIN_TEAMS,2)
            half=rng.sample(range(25,56),2);final=rng.sample(range(65,106),2)
            sig=tuple(half+final)
            if sig not in seen:seen.add(sig);break
        match=dict(id=f'joint-fresh-{i:03d}',teams=teams,half=half,final=final)
        for kind in ('winner','halftime','comeback'):
            rows.append(example(match,kind))
    out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest=dict(seed=20261107,matches=60,questions=len(rows),data_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        excluded_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        protocol_sha256=hashlib.sha256((HERE/'QUALITY_PROTOCOL.md').read_bytes()).hexdigest())
    (HERE/'quality-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='excluded_sources'},indent=2))


if __name__=='__main__':main()

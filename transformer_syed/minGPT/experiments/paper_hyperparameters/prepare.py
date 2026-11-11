"""Create a new immutable source/data snapshot before any study jobs run."""
import argparse
import datetime
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.paper_hyperparameters.common import ROOT,save,sha
from experiments.paper_hyperparameters.protocol import DESIGN
from experiments.sports.data import TRAIN_TEAMS,example


def prepare(study):
    study.mkdir(parents=True,exist_ok=False)
    repo=study/'snapshot/repo'
    sources=[ROOT/'lora_experiment.py']+list((ROOT/'mingpt').glob('*.py'))
    sources += [ROOT/'experiments/sports'/n for n in ('data.py','scoring.py','train.jsonl','validation.jsonl','test.jsonl','counterfactual.jsonl','manifest.json')]
    sources += [p for p in (ROOT/'experiments/paper_hyperparameters').iterdir() if p.is_file() and p.suffix in ('.py','.md','.sh','.bib')]
    for source in sources:
        target=repo/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    data=repo/'experiments/paper_hyperparameters/data';data.mkdir(parents=True)
    old=ROOT/'experiments/sports'
    shutil.copy2(old/'train.jsonl',data/'train.jsonl')
    # Exclude all old match signatures, including previously examined pilot test matches.
    seen=set()
    for name in ('train','validation','test','counterfactual'):
        for line in (old/(name+'.jsonl')).read_text().splitlines():
            m=json.loads(line)['match'];seen.add(tuple(m['half']+m['final']))
    rng=random.Random(DESIGN['data']['generator_seed']);splits={}
    for split,count,teams in [('validation',200,TRAIN_TEAMS),('test',300,DESIGN['data']['heldout_teams'])]:
        rows=[]
        for i in range(count):
            while True:
                a,b=rng.sample(teams,2);half=rng.sample(range(25,56),2);final=rng.sample(range(65,106),2)
                signature=tuple(half+final)
                if signature not in seen:seen.add(signature);break
            m=dict(id=f'paper-{split}-{i:04d}',teams=[a,b],half=half,final=final)
            for kind in ('winner','halftime','comeback'):
                row=example(m,kind);row['group_id']=m['id'];rows.append(row)
        splits[split]=rows
    counter=[]
    for row in splits['test']:
        m=dict(row['match']);field='final' if row['kind']=='winner' else 'half'
        m[field]=list(reversed(m[field]));m['id']+='-counter-'+row['kind']
        cf=example(m,row['kind']);cf['group_id']=row['group_id'];cf['pair_id']=row['match_id']+':'+row['kind'];counter.append(cf)
    splits['counterfactual']=counter
    for split,rows in splits.items():
        (data/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    all_rows={name:[json.loads(x) for x in (data/(name+'.jsonl')).read_text().splitlines()] for name in ('train','validation','test')}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not {r['question'] for r in all_rows[a]}&{r['question'] for r in all_rows[b]}
        assert not {tuple(r['match']['half']+r['match']['final']) for r in all_rows[a]}&{tuple(r['match']['half']+r['match']['final']) for r in all_rows[b]}
    assert not set(TRAIN_TEAMS)&set(DESIGN['data']['heldout_teams'])
    save(study/'protocol.json',DESIGN)
    manifest=dict(study_id=study.name,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  protocol_sha256=sha(study/'protocol.json'),
                  snapshot_sha256={str(p.relative_to(repo)):sha(p) for p in repo.rglob('*') if p.is_file()},
                  row_counts={k:len(v) for k,v in all_rows.items()}|{'counterfactual':len(counter)},
                  no_pilot_result_pooling=True,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(study/'manifest.json',manifest)
    (study/'logs').mkdir();print(study)
    return study


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path)
    prepare(p.parse_args().study.resolve())

"""Read durable artifacts and Slurm state without altering the experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.paper_hyperparameters.common import ROOT,read


def status(study):
    jobs=read(study/'jobs.json')['jobs'];ids=','.join(j['id'] for j in jobs.values())
    print('Study:',study)
    try:
        subprocess.run(['squeue','-j',ids,'-o','%.22i %.10T %.12M %.25j %R'],check=False,timeout=30)
    except subprocess.TimeoutExpired:print('Slurm queue query timed out; artifact status follows.')
    for name,pattern in [('search training','search-training/*/seed-*/training.json'),
                         ('runtime search','runtime-search/session-*.json'),
                         ('confirmation training','confirmation-training/*/seed-*/training.json'),
                         ('confirmation timing','confirmation-timing/*.json'),
                         ('quality','confirmation-quality/*.json')]:
        rows=[read(p) for p in study.glob(pattern)]
        print(name,':',sum(r.get('complete',False) for r in rows),'complete;',len(rows),'started')
    for p in sorted((study/'failures').glob('*.json')) if (study/'failures').exists() else []:
        r=read(p);print('FAILURE',p.name,r['error'])
    if (study/'selection.json').exists():
        s=read(study/'selection.json');print('Selected:',{k:s[k] for k in ('lora','chunk','tiles')})
    print('Final results:', 'ready' if (study/'results.json').exists() else 'pending')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',nargs='?',type=Path);args=p.parse_args()
    status(args.study.resolve() if args.study else Path(read(ROOT/'experiments/paper_hyperparameters/LATEST.json')['study']))

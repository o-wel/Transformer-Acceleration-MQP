"""Read Slurm state and real completed artifacts; never equate submission with completion."""
import argparse
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('run',type=Path,nargs='?')
args=p.parse_args()
root=args.run.resolve() if args.run else sorted((HERE/'runs').glob('*/jobs.json'),key=lambda f:f.stat().st_mtime)[-1].parent
jobs=json.loads((root/'jobs.json').read_text())
ids=','.join(r['job'] for r in jobs)
print(subprocess.check_output(['sacct','-j',ids,'--format=JobID,State,ExitCode,Elapsed,NodeList','-P'],text=True))
for name in ['basketball_fp16','spam_fp16','spam_fp32']:
    run=root/name
    trials=[json.loads(f.read_text()) for f in (run/'trials').glob('*/result.json')]
    cells=sum([json.loads(f.read_text()) for f in run.glob('systems-*.json')],[])
    smoke=run/'smoke.json'
    print(name,dict(preflight_passed=smoke.exists() and json.loads(smoke.read_text()).get('complete',False),
                    completed_trials=sum(r.get('complete',False) for r in trials),serving_cells=len(cells),
                    passing_cells=sum(r['status']=='ok' for r in cells)))
print('Report:',root/'PERFORMANCE_REVIEW.md')

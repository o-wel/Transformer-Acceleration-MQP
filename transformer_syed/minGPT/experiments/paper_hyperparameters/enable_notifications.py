"""Enable user-requested Slurm completion/failure email for an existing study."""
import argparse
import datetime
from email.utils import parseaddr
import json
import getpass
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.paper_hyperparameters.common import ROOT,read,save


def enable(study,address,dry_run=False):
    if '\n' in address or '\r' in address or parseaddr(address)[1]!=address or '@' not in address:
        raise ValueError('Provide a single valid destination email address')
    jobs=read(study/'jobs.json')['jobs'];updates=[]
    live=set(subprocess.check_output(['squeue','--noheader','--user',getpass.getuser(),'--format=%F'],text=True).split())
    for name,job in jobs.items():
        if job['id'] not in live:
            updates.append(dict(stage=name,job=job['id'],status='already finished; no update'))
            continue
        mail_type='END,FAIL' if name=='report' else 'FAIL'
        command=['scontrol','update',f'JobId={job["id"]}',f'MailUser={address}',f'MailType={mail_type}']
        if not dry_run:subprocess.run(command,check=True)
        updates.append(dict(stage=name,job=job['id'],mail_type=mail_type))
    if not dry_run:
        save(study/'notifications.json',dict(email=address,updates=updates,configured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    return updates


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path)
    parser.add_argument('--email',required=True);parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    study=args.study.resolve() if args.study else Path(read(ROOT/'experiments/paper_hyperparameters/LATEST.json')['study'])
    print(json.dumps(enable(study,args.email,args.dry_run),indent=2))

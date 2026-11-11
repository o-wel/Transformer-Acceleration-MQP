"""Submit the dependency DAG with an atomic, inspectable job ledger."""
import argparse
import datetime
from pathlib import Path
import shlex
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.paper_hyperparameters.common import ROOT,read,save,verify


def submit(study):
    verify(study)
    if (study/'jobs.json').exists():raise RuntimeError('Job ledger exists; inspect it before submitting duplicates')
    script=study/'snapshot/repo/experiments/paper_hyperparameters/run.py'
    jobs={};save(study/'jobs.json',dict(jobs=jobs,complete_submission=False))
    def job(name,stage,dependency=None,array=None,gpu=True,hours=12,qos='long'):
        # shlex.quote is shell quoting; user strings are never interpolated unquoted.
        body='\n'.join(['#!/usr/bin/env bash','set -eo pipefail','source "$HOME/MQP/Torch/setup.sh"',
                         'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0',
                         'export TOKENIZERS_PARALLELISM=false',
                         f'cd {shlex.quote(str(script.parents[2]))}',
                         f'python -u {shlex.quote(str(script))} {shlex.quote(stage)} --study {shlex.quote(str(study))} --index "${{SLURM_ARRAY_TASK_ID:-0}}"'])+'\n'
        path=study/'logs'/f'{name}.sh';path.write_text(body)
        args=['sbatch','--parsable','--account=isp','--partition=academic',f'--qos={qos}',
              f'--time={hours}:00:00','--cpus-per-task=4',f'--mem={32 if gpu else 8}G',
              f'--job-name=paper-{name}',f'--output={study}/logs/{name}-%A_%a.log','--kill-on-invalid-dep=yes']
        if gpu:args+=['--gres=gpu:A30:1']
        if dependency:args+=['--dependency=afterok:'+':'.join(dependency)]
        if array:args+=['--array='+array]
        args+=[str(path)]
        ident=subprocess.check_output(args,text=True).strip().split(';')[0]
        jobs[name]=dict(id=ident,stage=stage,dependency=dependency,array=array,gpu=gpu,time_limit_hours=hours,
                        command=args,script=str(path),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(study/'jobs.json',dict(jobs=jobs,complete_submission=False))
        print(name,ident,flush=True);return ident
    preflight=job('preflight','preflight',hours=1,qos='short')
    train=job('search-train','search-train',[preflight],array='0-44%3',hours=3)
    runtime=job('search-runtime','search-runtime',[train],array='0-2%3',hours=8)
    selection=job('select','select',[runtime],gpu=False,hours=1)
    confirm=job('confirm-train','confirm-train',[selection],array='0-4%2',hours=3)
    timing=job('confirm-runtime','confirm-runtime',[confirm],array='0-14%3',hours=12)
    quality=job('quality','quality',[confirm],array='0-4%1',hours=12)
    report=job('report','report',[timing,quality],gpu=False,hours=3)
    save(study/'jobs.json',dict(jobs=jobs,complete_submission=True))
    save(ROOT/'experiments/paper_hyperparameters/LATEST.json',dict(study=str(study),jobs={name:j['id'] for name,j in jobs.items()}))
    return jobs


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path)
    submit(p.parse_args().study.resolve())

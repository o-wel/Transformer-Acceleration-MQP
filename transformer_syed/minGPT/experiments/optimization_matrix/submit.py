"""Freeze a new study, record submissions immediately, and serialize task/precision studies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ENGINE=ROOT/'experiments/flash_attention/results/fp16_hyperparameter_search'
sys.path.insert(0,str(HERE))
from protocol import SEEDS,WORKLOADS,uid


def write(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)


def freeze(out):
    out.mkdir(parents=True,exist_ok=False)
    repo=out/'snapshot/repo';repo.mkdir(parents=True)
    files=list((ROOT/'mingpt').rglob('*.py'))+[ROOT/'lora_experiment.py',ROOT/'sports_experiment.py']
    files+=list(ENGINE.glob('*.py'))+list(HERE.glob('*.py'))+[HERE/'run.sh']
    for directory,names in [('experiments/sports',['scoring.py','train.jsonl','validation.jsonl','test.jsonl']),
                            ('experiments/kv_cache',['benchmark.py']),
                            ('experiments/sms_spam',['experiment.py','data/train.jsonl','data/validation.jsonl','data/test.jsonl','data/manifest.json'])]:
        files += [ROOT/directory/name for name in names]
    hashes={}
    for src in files:
        rel=src.relative_to(ROOT);dest=repo/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        hashes[str(rel)]=hashlib.sha256(dest.read_bytes()).hexdigest()
    old=ENGINE/'runs/20261111_fp16_matched'
    for task,precision in [('basketball','fp16'),('basketball','fp32'),('spam','fp16'),('spam','fp32')]:
        reference=ENGINE.parent/'hyperparameter_search/run_fp32' if (task,precision)==('basketball','fp32') else old/f'{task}_{precision}'
        selection=json.loads((reference/'selection.json').read_text())
        trials=[json.loads(f.read_text()) for f in (reference/'trials').glob('*/result.json')]
        eligible=[r for r in trials if r.get('complete') and r['examples']==9600 and r['seed']==42 and r['config']['method']=='lora']
        assert eligible
        worst=max(eligible,key=lambda r:(r['validation_nll'],uid(r['config'])))
        plan=dict(task=task,precision=precision,seeds=SEEDS,examples=9600,workloads=WORKLOADS,
            configs=dict(off=selection['full_control'],best=selection['winner'],poor=worst['config']),
            prior_source=str(reference),prior_selection_sha256=hashlib.sha256((reference/'selection.json').read_bytes()).hexdigest(),
            poor_trial=worst,selection='Frozen prior validation choices; poor uses worst seed-42 full-budget NLL',
            factorial_states=27,joint_extra_states=4,calibration_repeats=3,confirmation_repeats=7,
            precision_policy='FP16 autocast with FP32 master weights and stable reductions' if precision=='fp16' else 'FP32 with TF32 disabled')
        run=out/f'{task}_{precision}';run.mkdir();(run/'logs').mkdir()
        write(run/'plan.json',plan)
        dest=repo/'experiments/optimization_matrix/plans'/f'{run.name}.json';dest.parent.mkdir(exist_ok=True)
        shutil.copy2(run/'plan.json',dest);hashes[str(dest.relative_to(repo))]=hashlib.sha256(dest.read_bytes()).hexdigest()
    write(out/'SNAPSHOT_SHA256.json',hashes)
    write(out/'jobs.json',[])
    return repo/'experiments/optimization_matrix'


def submit(out,matrix,smoke_only=False):
    jobs=json.loads((out/'jobs.json').read_text());assert not jobs,'Refusing duplicate/partial submission; inspect jobs.json'
    previous=None
    for task,precision in [('basketball','fp16'),('basketball','fp32'),('spam','fp16'),('spam','fp32')]:
        run=out/f'{task}_{precision}'
        env=dict(os.environ,MQP_RUN=str(run),MQP_MATRIX=str(matrix))
        def job(stage,dependency=None,array=None,time='24:00:00',cpu=False,after='afterok'):
            command=['sbatch','--parsable','--kill-on-invalid-dep=yes',f'--time={time}',f'--job-name=matrix-{task}-{precision}-{stage}',
                     f'--output={run}/logs/{stage}-%A_%a.log']
            if cpu:command+=['--partition=cpu','--gres=none','--qos=short']
            if dependency:command+=['--dependency='+after+':'+dependency]
            if array:command+=['--array='+array]
            command += [str(matrix/'run.sh'),stage]
            ident=subprocess.check_output(command,env=env,text=True).strip().split(';')[0]
            jobs.append(dict(task=task,precision=precision,stage=stage,job=ident,dependency=dependency,array=array,run=str(run)))
            write(out/'jobs.json',jobs);print(task,precision,stage,ident,flush=True)
            return ident
        smoke=job('smoke',previous,time='00:30:00')
        if smoke_only:previous=smoke;continue
        prepare=job('prepare',smoke,time='08:00:00')
        calibration=job('calibrate',prepare,array='0-2%2')
        # CPU-light selection/report use a short GPU allocation on the known valid partition.
        selection=job('select',calibration,time='00:10:00')
        confirmation=job('confirm',selection,array='0-8%2')
        quality=job('quality',confirmation,array='0-2%2')
        previous=job('report',quality,time='00:20:00',after='afterany')
        subprocess.run([sys.executable,str(matrix/'report_cli.py'),str(run)],check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args();out=a.out.resolve();matrix=freeze(out);submit(out,matrix,a.smoke_only)


if __name__=='__main__':main()

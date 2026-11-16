"""Freeze sources/data and submit matched FP16 studies plus an SMS FP32 control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--report-deps',type=Path,required=True)
    args=p.parse_args();out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    repo=out/'snapshot/repo';repo.mkdir(parents=True)
    files=list((ROOT/'mingpt').rglob('*.py'))+[ROOT/'lora_experiment.py', ROOT/'sports_experiment.py']
    files+=list(HERE.glob('*.py'))+[HERE/'run_gpu.sh']
    for directory,names in [('experiments/sports',['scoring.py','train.jsonl','validation.jsonl','test.jsonl']),('experiments/kv_cache',['benchmark.py']),('experiments/sms_spam',['experiment.py','data/train.jsonl','data/validation.jsonl','data/test.jsonl','data/manifest.json'])]:
        files += [ROOT/directory/name for name in names]
    hashes={}
    for src in files:
        rel=src.relative_to(ROOT);dest=repo/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dest);hashes[str(rel)]=hashlib.sha256(dest.read_bytes()).hexdigest()
    (out/'SNAPSHOT_SHA256.json').write_text(json.dumps(hashes,indent=2))
    # Copy only original FP32 measured artifacts, never model checkpoints.
    reference=out/'fp32_basketball_reference';reference.mkdir()
    old=HERE.parent/'hyperparameter_search/run_fp32'
    for src in list(old.glob('*.json'))+list((old/'trials').glob('*/result.json')):
        dest=reference/src.relative_to(old);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    frozen=repo/HERE.relative_to(ROOT)
    env=dict(os.environ,MQP_FROZEN=str(frozen),MQP_SNAPSHOT=str(out),MQP_REPORT_DEPS=str(args.report_deps.resolve()),PYTHONPATH=str(args.report_deps.resolve()))
    jobs=[]
    def submit(stage,run,task,precision,dependency=None,array=None,time='08:00:00'):
        run.mkdir(exist_ok=True);(run/'logs').mkdir(exist_ok=True)
        jobenv=dict(env,MQP_RUN=str(run),MQP_TASK=task,MQP_PRECISION=precision)
        command=['sbatch','--parsable',f'--time={time}',f'--output={run}/logs/{stage}-%A_%a.log']
        if dependency:command+=['--dependency=afterok:'+dependency]
        if array:command+=['--array='+array]
        command += [str(frozen/'run_gpu.sh'),stage]
        job=subprocess.check_output(command,env=jobenv,text=True).strip().split(';')[0]
        jobs.append(dict(stage=stage,task=task,precision=precision,job=job,dependency=dependency,array=array,run=str(run)))
        (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
        print(task,precision,stage,job,flush=True)
        return job
    previous=None
    # Serial studies; screen/systems arrays permit two concurrent GPU jobs, as in FP32.
    for task,precision in [('basketball','fp16'),('spam','fp16'),('spam','fp32')]:
        run=out/f'{task}_{precision}';run.mkdir()
        subprocess.run([sys.executable,str(frozen/'study.py'),'plan','--out',str(run),'--task',task,'--precision',precision],check=True)
        smoke=submit('smoke',run,task,precision,previous,time='00:30:00')
        screen=submit('screen',run,task,precision,smoke,array='0-2%2')
        refine=submit('refine',run,task,precision,screen)
        regularize=submit('regularize',run,task,precision,refine)
        batch=submit('batch',run,task,precision,regularize)
        confirm=submit('confirm',run,task,precision,batch)
        systems=submit('systems',run,task,precision,confirm,array='0-2%2',time='24:00:00')
        previous=submit('report',run,task,precision,systems,time='00:20:00')
        # Generate honest initial reports before measurements exist.
        subprocess.run([sys.executable,str(frozen/'build_report.py'),'--out',str(run)],check=True,env=env)
    subprocess.run([sys.executable,str(frozen/'compare.py'),str(out)],check=True,env=env)


if __name__=='__main__':main()

"""Artifact IO and frozen-study integrity checks; safe to import without torch."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def save(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify(study):
    m=read(study/'manifest.json')
    if sha(study/'protocol.json')!=m['protocol_sha256']:raise RuntimeError('Frozen protocol changed')
    if (study/'selection.sha256').exists() and sha(study/'selection.json')!=(study/'selection.sha256').read_text().strip():raise RuntimeError('Frozen selection changed')
    for rel,expected in m['snapshot_sha256'].items():
        if sha(study/'snapshot/repo'/rel)!=expected:raise RuntimeError(f'Frozen source/data changed: {rel}')
    return m


def environment():
    import torch
    import transformers
    import triton
    import numpy
    info=dict(job=os.environ.get('SLURM_JOB_ID'),array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),
              node=platform.node(),python=platform.python_version(),torch=torch.__version__,
              transformers=transformers.__version__,triton=triton.__version__,numpy=numpy.__version__,cuda=torch.version.cuda,
              cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
              gpu=torch.cuda.get_device_name(),gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),
              started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    try:
        info['gpu_telemetry']=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout
    except (OSError,subprocess.TimeoutExpired):info['gpu_telemetry']='unavailable'
    return info


def configure(study):
    verify(study)
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    info=environment()
    if info['gpu']!='NVIDIA A30':raise RuntimeError('This registered study requires NVIDIA A30')
    preflight=study/'preflight.json'
    if preflight.exists():
        old=read(preflight)['environment']
        for k in ('python','torch','transformers','triton','numpy','cuda','gpu'):
            if info[k]!=old[k]:raise RuntimeError(f'Environment drift: {k}')
    return info

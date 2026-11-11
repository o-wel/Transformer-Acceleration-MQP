import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import torch
from mingpt.model import GPT
from experiments.joint_decode.benchmark import fingerprint,sha,save

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
STUDY=ROOT/'experiments/paper_hyperparameters/runs/study-20261105-v2'
GRID=[(bn,w) for bn in (32,64,128,256) for w in (1,2,4,8)]


def environment(seed,phase):
    torch.set_num_threads(1);torch.manual_seed(20261110+seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((HERE/'manifest.json').read_text())
    assert sha(HERE/'PROTOCOL.md')==manifest['protocol_sha256']
    assert sha(HERE/'fresh-quality.jsonl')==manifest['data_sha256']
    assert sha(STUDY/'selection.json')==manifest['inherited_selection_sha256']
    import triton,transformers,numpy
    return dict(complete=False,phase=phase,seed=seed,job=os.environ['SLURM_JOB_ID'],node=platform.node(),
        torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__,transformers=transformers.__version__,numpy=numpy.__version__,
        gpu=torch.cuda.get_device_name(),gpu_telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,clocks.sm,temperature.gpu','--format=csv'],text=True),
        protocol_sha256=manifest['protocol_sha256'],inherited_selection_sha256=manifest['inherited_selection_sha256'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'runtime.py',HERE/'common.py',HERE/'study.py',HERE/'PROTOCOL.md',
            HERE/'quality.py',ROOT/'mingpt/model.py',ROOT/'experiments/joint_decode/kernel.py',ROOT/'experiments/joint_decode/runtime.py']})


def load(seed,phase):
    config=json.loads((HERE/'manifest.json').read_text())['lora']
    part='search-training' if phase=='search' else 'confirmation-training'
    folder=STUDY/part/f"r{config['rank']}-a{config['alpha']}-lr{config['lr']:g}"/f'seed-{seed}'
    record=json.loads((folder/'training.json').read_text());checkpoint=folder/'adapter.pt'
    assert record['complete'] and record['config']==config and sha(checkpoint)==record['adapter_sha256']
    model=GPT.from_pretrained('gpt2').enable_lora(config['rank'],config['alpha'])
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if not p.requires_grad:h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    assert h.hexdigest()==record['frozen_base_before']==record['frozen_base_after']
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert set(state)=={n for n,p in model.named_parameters() if p.requires_grad}
    missing,unexpected=model.load_state_dict(state,strict=False)
    assert not unexpected and all('.A.' not in k and '.B.' not in k for k in missing)
    for m in model.modules():
        if isinstance(m,torch.nn.Dropout):m.p=0.
    model=model.eval().half().cuda();model.set_attention_backend('flash')
    provenance=dict(checkpoint=str(checkpoint.relative_to(ROOT)),adapter_sha256=record['adapter_sha256'],
        training_config=config,selected_training_step=record['selected_step'],validation_nll=record['validation_nll'],
        frozen_base_sha256=record['frozen_base_before'])
    return model,provenance


def references(model):
    cases=[]
    for b in (1,4):
        for p in (37,896):
            ids=torch.randint(50257,(b,p+3),device='cuda');cache=None;refs=[]
            for step in range(4):
                x=ids[:,:p] if step==0 else ids[:,p+step-1:p+step]
                logits,_,cache=model(x,use_cache=True,past_key_values=cache,last_token_only=True)
                refs.append(logits.float().clone())
            cases.append((b,p,ids,refs))
    return cases


def check_runner(runner,cases):
    rows=[];vals_by_case=[]
    for b,p,ids,refs in cases:
        if b!=runner.batch:continue
        runner.prefill(ids[:,:p]);values=[runner.prefill_logits.float().clone()]
        for step in range(3):
            runner.token.copy_(ids[:,p+step:p+step+1]);runner.replay();values.append(runner.step_logits.float().clone())
        for step,(got,ref) in enumerate(zip(values,refs)):
            torch.testing.assert_close(got,ref,atol=.1,rtol=.02)
            rows.append(dict(batch=b,prompt=p,step=step,max_error=float((got-ref).abs().max()),
                argmax_disagreements=int((got.argmax(-1)!=ref.argmax(-1)).sum())))
        vals_by_case.append(values)
    return rows,vals_by_case

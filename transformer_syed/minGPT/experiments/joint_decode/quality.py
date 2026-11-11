import json
import os
from pathlib import Path
import platform
import numpy as np
import torch
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from lora_experiment import prompt
from experiments.sports.scoring import score
from experiments.joint_decode.runtime import mode
from experiments.joint_decode.benchmark import CHECKPOINT, fingerprint, sha, save

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


@torch.inference_mode()
def evaluate(model,tok,row,name):
    prefix=tok.encode(prompt(row['question']))
    ids=torch.tensor([prefix],device='cuda')
    answer=tok.encode(' '+row['answer'])+[tok.eos_token_id]
    with mode(model,name):
        output=model.generate(ids,64,use_cache=True)[0,len(prefix):].tolist()
    clean=output[:output.index(tok.eos_token_id)] if tok.eos_token_id in output else output
    text=tok.decode(clean,skip_special_tokens=True).split('\nUser:')[0].strip()
    with mode(model,name):
        logits,_,cache=model(ids,use_cache=True,last_token_only=True)
        nll=0.
        for j,token in enumerate(answer):
            nll-=float(logits[0,-1].float().log_softmax(-1)[token])
            if j+1<len(answer):
                logits,_,cache=model(torch.tensor([[token]],device='cuda'),past_key_values=cache,
                                     use_cache=True,last_token_only=True)
    return dict(output_ids=output,text=text,scores=score(row,text),nll_sum=nll,answer_tokens=len(answer))


@torch.inference_mode()
def main():
    job=os.environ['SLURM_JOB_ID'];out=HERE/'results'/f'quality-{job}.json'
    if out.exists():raise RuntimeError('Refuse overwrite')
    manifest=json.loads((HERE/'quality-manifest.json').read_text())
    assert sha(HERE/'quality-data.jsonl')==manifest['data_sha256']
    assert sha(HERE/'QUALITY_PROTOCOL.md')==manifest['protocol_sha256']
    torch.set_num_threads(1);torch.manual_seed(20261107)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=GPT.from_pretrained('gpt2').enable_lora(4,8)
    missing,unexpected=model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True),strict=False)
    assert not unexpected and all('.A.' not in k and '.B.' not in k for k in missing)
    for m in model.modules():
        if isinstance(m,torch.nn.Dropout):m.p=0.
    model=model.eval().half().cuda()
    tok=GPT2Tokenizer.from_pretrained('gpt2',local_files_only=True)
    rows=[json.loads(x) for x in (HERE/'quality-data.jsonl').read_text().splitlines()]
    report=dict(complete=False,job=job,node=platform.node(),gpu=torch.cuda.get_device_name(),
        adapter_sha256=sha(CHECKPOINT),data_sha256=manifest['data_sha256'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'kernel.py',HERE/'runtime.py',
            HERE/'QUALITY_PROTOCOL.md',ROOT/'experiments/sports/scoring.py',ROOT/'mingpt/model.py']},
        weights_before=fingerprint(model),outputs=[])
    save(out,report)
    try:
        for i,row in enumerate(rows):
            methods=('dynamic_flash','fused_stream') if i%2==0 else ('fused_stream','dynamic_flash')
            pair={name:evaluate(model,tok,row,name) for name in methods}
            report['outputs'].append(dict(match_id=row['match_id'],kind=row['kind'],reference=row['answer'],methods=pair))
            if (i+1)%15==0:
                save(out,report);print('QUALITY',i+1,'/',len(rows),flush=True)
        summary={}
        for name in ('dynamic_flash','fused_stream'):
            vals=[r['methods'][name] for r in report['outputs']]
            summary[name]=dict(nll=sum(v['nll_sum'] for v in vals)/sum(v['answer_tokens'] for v in vals),
                **{k:sum(v['scores'][k] for v in vals)/len(vals) for k in vals[0]['scores']})
        summary['absolute_nll_difference']=abs(summary['dynamic_flash']['nll']-summary['fused_stream']['nll'])
        summary['nll_gate_passed']=summary['absolute_nll_difference']<=.01
        summary['exact_answer_agreement']=sum(r['methods']['dynamic_flash']['text']==r['methods']['fused_stream']['text'] for r in report['outputs'])
        summary['exact_raw_sequence_agreement']=sum(r['methods']['dynamic_flash']['output_ids']==r['methods']['fused_stream']['output_ids'] for r in report['outputs'])
        groups=sorted({r['match_id'] for r in report['outputs']})
        differences=[];discordant=0
        for group in groups:
            vals=[r for r in report['outputs'] if r['match_id']==group]
            differences.append(np.mean([int(r['methods']['fused_stream']['scores']['reference_exact'])-
                int(r['methods']['dynamic_flash']['scores']['reference_exact']) for r in vals]))
            discordant+=any(r['methods']['fused_stream']['text']!=r['methods']['dynamic_flash']['text'] for r in vals)
        differences=np.array(differences)
        rng=np.random.default_rng(20261107)
        boots=differences[rng.integers(0,len(groups),(5000,len(groups)))].mean(1)
        summary['accuracy_difference']=float(differences.mean())
        summary['accuracy_difference_cluster_ci95']=[float(v) for v in np.quantile(boots,[.025,.975])]
        summary['discordant_matches']=int(discordant)
        if discordant==0:summary['zero_discordance_match_probability_upper95']=1-.05**(1/len(groups))
        report['summary']=summary
        report['weights_after']=fingerprint(model)
        assert report['weights_before']==report['weights_after']
        report['complete']=True
    except Exception as e:
        report['failure']=repr(e);raise
    finally:save(out,report)
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':main()

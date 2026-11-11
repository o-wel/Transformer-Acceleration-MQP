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
from experiments.joint_decode.quality import evaluate
from experiments.joint_decode.benchmark import CHECKPOINT,fingerprint,sha,save
from experiments.graph_decode.runtime import Runner

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent


def graph_evaluate(runner,tok,row):
    prefix=tok.encode(prompt(row['question']));ids=torch.tensor([prefix],device='cuda')
    answer=tok.encode(' '+row['answer'])+[tok.eos_token_id]
    output=runner.generate(ids,64)[0,len(prefix):].tolist()
    clean=output[:output.index(tok.eos_token_id)] if tok.eos_token_id in output else output
    text=tok.decode(clean,skip_special_tokens=True).split('\nUser:')[0].strip()
    runner.prefill(ids);logits=runner.prefill_logits;nll=0.
    for j,token in enumerate(answer):
        nll-=float(logits[0,-1].float().log_softmax(-1)[token])
        if j+1<len(answer):
            runner.token.copy_(torch.tensor([[token]],device='cuda'));runner.replay();logits=runner.step_logits
    return dict(output_ids=output,text=text,scores=score(row,text),nll_sum=nll,answer_tokens=len(answer))


@torch.inference_mode()
def main():
    job=os.environ['SLURM_JOB_ID'];out=HERE/'results'/f'quality-{job}.json'
    if out.exists():raise RuntimeError('Refuse overwrite')
    manifest=json.loads((HERE/'quality-manifest.json').read_text())
    assert sha(HERE/'quality-data.jsonl')==manifest['data_sha256']
    assert sha(HERE/'CONFIRMATION_PROTOCOL.md')==manifest['protocol_sha256']
    torch.set_num_threads(1);torch.manual_seed(20261108)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=GPT.from_pretrained('gpt2').enable_lora(4,8)
    missing,unexpected=model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True),strict=False)
    assert not unexpected and all('.A.' not in k and '.B.' not in k for k in missing)
    for m in model.modules():
        if isinstance(m,torch.nn.Dropout):m.p=0.
    model=model.half().eval().cuda();model.set_attention_backend('flash')
    runner=Runner(model,1,True,True);tok=GPT2Tokenizer.from_pretrained('gpt2',local_files_only=True)
    rows=[json.loads(x) for x in (HERE/'quality-data.jsonl').read_text().splitlines()]
    sources=[Path(__file__),HERE/'runtime.py',HERE/'CONFIRMATION_PROTOCOL.md',ROOT/'mingpt/model.py',
        ROOT/'experiments/joint_decode/quality.py',ROOT/'experiments/joint_decode/runtime.py',ROOT/'experiments/joint_decode/kernel.py',
        ROOT/'experiments/sports/scoring.py']
    report=dict(complete=False,job=job,node=platform.node(),gpu=torch.cuda.get_device_name(),
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},adapter_sha256=sha(CHECKPOINT),
        data_sha256=manifest['data_sha256'],weights_before=fingerprint(model),outputs=[])
    save(out,report)
    try:
        for i,row in enumerate(rows):
            names=['original','previous_fusion','graph_fused'];names=names[i%3:]+names[:i%3]
            pair={}
            for name in names:
                if name=='graph_fused':pair[name]=graph_evaluate(runner,tok,row)
                else:pair[name]=evaluate(model,tok,row,'dynamic_flash' if name=='original' else 'fused_stream')
            report['outputs'].append(dict(match_id=row['match_id'],kind=row['kind'],reference=row['answer'],methods=pair))
            if (i+1)%30==0:save(out,report);print('QUALITY',i+1,'/ 360',flush=True)
        summary={}
        for name in ('original','previous_fusion','graph_fused'):
            vals=[r['methods'][name] for r in report['outputs']]
            summary[name]=dict(nll=sum(v['nll_sum'] for v in vals)/sum(v['answer_tokens'] for v in vals),
                **{k:sum(v['scores'][k] for v in vals)/len(vals) for k in vals[0]['scores']})
        groups=sorted({r['match_id'] for r in report['outputs']});rng=np.random.default_rng(20261108)
        picks=rng.integers(0,len(groups),(5000,len(groups)))
        summary['comparisons']={}
        for name in ('previous_fusion','graph_fused'):
            comp=dict(absolute_nll_difference=abs(summary[name]['nll']-summary['original']['nll']),
                exact_answer_agreement=sum(r['methods'][name]['text']==r['methods']['original']['text'] for r in report['outputs']),
                exact_raw_sequence_agreement=sum(r['methods'][name]['output_ids']==r['methods']['original']['output_ids'] for r in report['outputs']))
            diffs=[];discordant=0
            for group in groups:
                vals=[r for r in report['outputs'] if r['match_id']==group]
                diffs.append(np.mean([int(r['methods'][name]['scores']['reference_exact'])-
                    int(r['methods']['original']['scores']['reference_exact']) for r in vals]))
                discordant+=any(r['methods'][name]['text']!=r['methods']['original']['text'] for r in vals)
            diffs=np.array(diffs);boots=diffs[picks].mean(1)
            comp.update(nll_gate_passed=comp['absolute_nll_difference']<=.01,accuracy_difference=float(diffs.mean()),
                accuracy_difference_cluster_ci95=[float(v) for v in np.quantile(boots,[.025,.975])],discordant_matches=int(discordant))
            if discordant==0:comp['zero_discordance_probability_upper95']=1-.05**(1/len(groups))
            summary['comparisons'][name]=comp
        report['summary']=summary;report['weights_after']=fingerprint(model)
        assert report['weights_before']==report['weights_after'];report['complete']=True
    except Exception as e:
        report['failure']=repr(e);raise
    finally:save(out,report)
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':main()

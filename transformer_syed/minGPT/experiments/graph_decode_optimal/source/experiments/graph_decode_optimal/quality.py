import json
from collections import defaultdict
import torch
from transformers import GPT2Tokenizer
from lora_experiment import prompt
from experiments.sports.scoring import score
from experiments.graph_decode_optimal.runtime import Runner
from experiments.graph_decode_optimal.common import HERE,sha


def run(model,merged,runners,selected):
    tok=GPT2Tokenizer.from_pretrained('gpt2',local_files_only=True)
    rows=[json.loads(s) for s in (HERE/'fresh-quality.jsonl').read_text().splitlines()]
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        prefix=tok.encode(prompt(row['question']));groups[len(prefix)].append((i,prefix,row))
    output=[None]*len(rows);extra={}
    def get_runner(which,b):
        name='selected_graph' if which=='selected_graph' else 'merged_graph'
        if b in runners:return runners[b][name]
        key=(name,b)
        if key not in extra:extra[key]=Runner(model if name=='selected_graph' else merged,b,True if name=='selected_graph' else False,
            True,bn=selected[0],warps=selected[1])
        return extra[key]
    for length,group in sorted(groups.items()):
        for start in range(0,len(group),4):
            part=group[start:start+4];ids=torch.tensor([p for _,p,_ in part],device='cuda')
            methods={}
            for name in ('original','selected_graph','merged_graph'):
                if name=='original':got=model.generate(ids,64,use_cache=True)
                else:got=get_runner(name,len(part)).generate(ids,64)
                methods[name]=got[:,length:].tolist()
            for k,(index,prefix,row) in enumerate(part):
                results={}
                for name in methods:
                    raw=methods[name][k];clean=raw[:raw.index(tok.eos_token_id)] if tok.eos_token_id in raw else raw
                    text=tok.decode(clean,skip_special_tokens=True).split('\nUser:')[0].strip()
                    results[name]=dict(output_ids=raw,text=text,scores=score(row,text))
                output[index]=dict(match_id=row['match_id'],kind=row['kind'],reference=row['answer'],methods=results)
        print('QUALITY LENGTH BUCKET',length,'DONE',flush=True)
    # Fixed answer-token likelihood; each runtime sees identical teacher inputs.
    totals={n:[0.,0] for n in ('original','selected_graph','merged_graph')}
    for index,row in enumerate(rows):
        ids=torch.tensor([tok.encode(prompt(row['question']))],device='cuda')
        targets=tok.encode(' '+row['answer'])+[tok.eos_token_id]
        for name in totals:
            if name=='original':
                logits,_,cache=model(ids,use_cache=True,last_token_only=True)
            else:
                runner=get_runner(name,1);runner.prefill(ids);logits=runner.prefill_logits
            nll=0.
            for j,target in enumerate(targets):
                nll-=float(logits[0,-1].float().log_softmax(-1)[target])
                if j+1<len(targets):
                    x=torch.tensor([[target]],device='cuda')
                    if name=='original':logits,_,cache=model(x,past_key_values=cache,use_cache=True,last_token_only=True)
                    else:runner.token.copy_(x);runner.replay();logits=runner.step_logits
            totals[name][0]+=nll;totals[name][1]+=len(targets)
            output[index]['methods'][name].update(nll_sum=nll,answer_tokens=len(targets))
        if (index+1)%90==0:print('QUALITY NLL',index+1,'/360',flush=True)
    summary={}
    for name,(nll,count) in totals.items():
        summary[name]=dict(nll=nll/count,**{key:sum(r['methods'][name]['scores'][key] for r in output)/len(output)
            for key in output[0]['methods'][name]['scores']})
    summary['comparisons']={}
    for name in ('selected_graph','merged_graph'):
        delta=abs(summary[name]['nll']-summary['original']['nll'])
        summary['comparisons'][name]=dict(absolute_nll_difference=delta,nll_gate_passed=delta<=.01,
            exact_answer_agreement=sum(r['methods'][name]['text']==r['methods']['original']['text'] for r in output),
            raw_sequence_agreement=sum(r['methods'][name]['output_ids']==r['methods']['original']['output_ids'] for r in output))
    return dict(data_sha256=sha(HERE/'fresh-quality.jsonl'),summary=summary,outputs=output)

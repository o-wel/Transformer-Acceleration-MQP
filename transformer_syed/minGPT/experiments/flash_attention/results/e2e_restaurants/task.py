"""E2E answer-only supervision and upstream semantic-slot scoring.

The scorer is a heuristic, not a human-verified assertion of arbitrary factuality.
"""
import ast
import importlib.util
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('e2e_tgen_data',HERE/'vendor/tgen_data.py')
DA_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(DA_module)
DA=DA_module.DA
# Execute the unchanged upstream pattern engine; omit only CLI/dataframe I/O.
tree=ast.parse((HERE/'vendor/slot_error.py').read_text())
nodes=[]
for node in tree.body:
    if isinstance(node,ast.Import) and any(a.name=='pandas' for a in node.names):continue
    if isinstance(node,ast.ImportFrom) and node.module=='tgen.data':continue
    if isinstance(node,ast.If):continue
    if isinstance(node,ast.FunctionDef) and node.name in ('process_file','load_lines'):continue
    nodes.append(node)
SCORER={'DA':DA,'__name__':'e2e_upstream_scorer'}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned E2E slot_error.py','exec'),SCORER)
MAX_NEW_TOKENS=96


def canonical(mr):
    return sorted((d.slot,d.value) for d in DA.parse_diligent_da(mr).dais)


def prompt(row):
    return 'Describe this restaurant using all the facts below. Do not add facts.\nFacts: '+row['mr']+'\nDescription:'


def score(mr,text):
    parsed=DA.parse_diligent_da(mr)
    predicted,gold=SCORER['reclassify_mr'](text,parsed)
    added,missing,wrong,repeated,diff,_=SCORER['check_output'](gold,predicted)
    n=len(parsed)
    return dict(added=added,missing=missing,wrong=wrong,repeated=repeated,slots=n,
                slot_error_rate=(added+missing+wrong+repeated)/n,
                semantic_exact=int(added+missing+wrong+repeated==0),
                slot_coverage=max(0.,1-(missing+wrong)/n),diff=diff)


def aggregate(outputs):
    total=sum(o['scores']['slots'] for o in outputs)
    result={k:sum(o['scores'][k] for o in outputs)/total for k in ('added','missing','wrong','repeated')}
    result.update(slot_error_rate=sum(sum(o['scores'][k] for k in ('added','missing','wrong','repeated')) for o in outputs)/total,
                  semantic_exact=statistics.mean(o['scores']['semantic_exact'] for o in outputs),
                  slot_coverage=statistics.mean(o['scores']['slot_coverage'] for o in outputs),
                  termination_rate=statistics.mean(o['terminated'] for o in outputs),
                  mean_output_tokens=statistics.mean(len(o['tokens']) for o in outputs))
    return result


def encode(rows,tokenizer,length,torch):
    result=[]
    for row in rows:
        p=tokenizer.encode(prompt(row));a=tokenizer.encode(' '+row['reference'])+[tokenizer.eos_token_id]
        if len(p)+len(a)>length:raise ValueError('Refusing to truncate restaurant facts or references')
        ids=p+a
        x=torch.full((length,),tokenizer.eos_token_id,dtype=torch.long)
        y=torch.full((length,),-1,dtype=torch.long)
        x[:len(ids)]=torch.tensor(ids)
        y[len(p)-1:len(ids)-1]=torch.tensor(a)
        result.append((x,y))
    return result


def evaluate(model,rows,tokenizer,torch,config=None,configure=None):
    outputs=[];model.eval()
    for row in rows:
        ids=tokenizer.encode(prompt(row))
        if config is not None:
            configure(model,config,dict(batch=1,prompt_tokens=len(ids),new_tokens=MAX_NEW_TOKENS))
        x=torch.tensor([ids],device='cuda')
        with torch.inference_mode():
            seq=model.generate(x,MAX_NEW_TOKENS,use_cache=config is None or config['cache_mode']!='off')[0,len(ids):].tolist()
        terminated=tokenizer.eos_token_id in seq
        if terminated:seq=seq[:seq.index(tokenizer.eos_token_id)]
        text=tokenizer.decode(seq,skip_special_tokens=True).strip()
        outputs.append(dict(id=row['id'],mr=row['mr'],text=text,tokens=seq,terminated=terminated,
                            scores=score(row['mr'],text)))
    return aggregate(outputs),outputs

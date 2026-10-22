"""Post-hoc vocabulary control: unchanged held-out scores, familiar team names."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from sports_experiment import evaluate_outputs,load_parameters
from lora_experiment import load_rows,scales
from experiments.sports.data import example


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    manifest=json.loads(Path('experiments/sports/matched/selection.json').read_text())
    assert manifest.get('complete')
    mapping={'Bulls':'Bears','Eagles':'Comets','Owls':'Falcons','Sharks':'Hawks'}
    originals=load_rows('experiments/sports/test.jsonl')
    rows=[]
    for row in originals:
        match=dict(row['match'])
        match['teams']=[mapping[t] for t in match['teams']]
        rows.append(example(match,row['kind']))
    trained={r['answer'] for r in load_rows('experiments/sports/train.jsonl')}
    report=dict(exploratory=True,protocol='Added after observing seed-42 unseen-team failures. Replace names only; keep original held-out scores/questions. No retraining or tuning. This does not replace the original test.',mapping=mapping,reference_answers_seen_in_training=sum(r['answer'] in trained for r in rows),gpu=torch.cuda.get_device_name(),methods={})
    Path('experiments/sports/matched/familiar-control.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
    a=argparse.Namespace(batch_size=8,device='cuda',new_tokens=40)
    for method in ['lora','full']:
        report['methods'][method]={}
        for seed,path in manifest['methods'][method]['runs'].items():
            model=GPT.from_pretrained('gpt2').float()
            ckpt=torch.load(Path(path)/('adapter.pt' if method=='lora' else 'model.pt'),map_location='cpu',weights_only=True)
            if method=='lora': model.enable_lora(ckpt['rank'],ckpt['alpha'])
            load_parameters(model,ckpt)
            del ckpt
            model.to('cuda').eval()
            if method=='lora' and seed=='42':
                scales(model,False)
                report['base']=evaluate_outputs(model,tokenizer,rows,a)
                scales(model,True)
            result=evaluate_outputs(model,tokenizer,rows,a)
            report['methods'][method][seed]=result
            print(method,seed,json.dumps(result['metrics']),flush=True)
            del model
            torch.cuda.empty_cache()
    Path('experiments/sports/matched/familiar-control.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()

"""Live pretrained/full-fine-tuning/LoRA comparison on fictional match scores."""
import argparse
import json
from pathlib import Path
import torch
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from lora_experiment import scales
from sports_experiment import generate_batch, load_parameters
from experiments.sports.data import example


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter',default='out/sports-seed42/adapter.pt')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--matched',action='store_true',help='Use the validation-selected matched full/LoRA models')
    parser.add_argument('--teams',nargs=2,default=['Eagles','Owls'])
    parser.add_argument('--half',nargs=2,type=int,default=[30,40])
    parser.add_argument('--final',nargs=2,type=int,default=[85,80])
    parser.add_argument('--question',choices=['winner','halftime','comeback'],default='comeback')
    parser.add_argument('--interactive',action='store_true')
    a=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    full_model=None
    if a.matched:
        selection=json.loads(Path('experiments/sports/matched/selection.json').read_text())
        if not selection.get('complete'):
            raise RuntimeError('Matched experiment is not complete')
        a.adapter=str(Path(selection['methods']['lora']['runs']['42'])/'adapter.pt')
        full_model=GPT.from_pretrained('gpt2').float()
        full_ckpt=torch.load(Path(selection['methods']['full']['runs']['42'])/'model.pt',map_location='cpu',weights_only=True)
        load_parameters(full_model,full_ckpt)
        full_model.to(a.device).eval()
        del full_ckpt
    ckpt=torch.load(a.adapter,map_location='cpu',weights_only=True)
    model=GPT.from_pretrained('gpt2').float().enable_lora(ckpt['rank'],ckpt['alpha'])
    load_parameters(model,ckpt)
    model.to(a.device).eval()
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
    while True:
        if a.interactive:
            try:
                team_line=input('Two team names, comma-separated (blank to quit): ').strip()
                if not team_line: break
                a.teams=[x.strip() for x in team_line.split(',')]
                a.half=list(map(int,input('Halftime scores in team order: ').split()))
                a.final=list(map(int,input('Final scores in team order: ').split()))
                a.question=input('Question: winner / halftime / comeback: ').strip()
            except EOFError:
                break
            except ValueError:
                print('Enter integer scores.'); continue
        if len(a.teams)!=2 or len(a.half)!=2 or len(a.final)!=2 or a.teams[0]==a.teams[1] or a.question not in ['winner','halftime','comeback']:
            print('Need two distinct teams, two scores per period, and a supported question.')
            if a.interactive: continue
            raise SystemExit(2)
        if a.half[0]==a.half[1] or a.final[0]==a.final[1] or any(h<0 or f<h for h,f in zip(a.half,a.final)):
            print('Use non-tied scores with final scores at least as large as halftime scores.')
            if a.interactive: continue
            raise SystemExit(2)
        row=example(dict(id='live',teams=a.teams,half=a.half,final=a.final),a.question)
        print('\n'+row['question'])
        if full_model is not None:
            scales(model,False)
            print('PRETRAINED: '+generate_batch(model,tokenizer,[row['question']],a.device,40)[0])
            print('FULL FINE-TUNING: '+generate_batch(full_model,tokenizer,[row['question']],a.device,40)[0])
        for enabled,name in ([(True,'LORA')] if full_model is not None else [(False,'BASE'),(True,'LORA')]):
            scales(model,enabled)
            print(name+': '+generate_batch(model,tokenizer,[row['question']],a.device,40)[0])
        print('REFERENCE: '+row['answer']+'\n')
        if not a.interactive: break


if __name__=='__main__':
    main()

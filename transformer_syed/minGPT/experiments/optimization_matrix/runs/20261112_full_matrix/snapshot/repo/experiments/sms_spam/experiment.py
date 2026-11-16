"""SMS spam classification: GPT-2 base, full fine-tuning, and LoRA; no sports data."""
import argparse
import hashlib
import io
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
LABELS = ['ham', 'spam']
URL = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def template(text):
    text = re.sub(r'(?:https?://|www\.)\S+', ' URL ', text.lower())
    text = re.sub(r'\d+', ' NUM ', text)
    return re.sub(r'\W+', ' ', text).strip().lower()


def metrics(gold, pred):
    tp = sum(y == 1 and p == 1 for y, p in zip(gold, pred))
    tn = sum(y == 0 and p == 0 for y, p in zip(gold, pred))
    fp = sum(y == 0 and p == 1 for y, p in zip(gold, pred))
    fn = sum(y == 1 and p == 0 for y, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    specificity = tn / (tn + fp) if tn + fp else 0.
    return dict(accuracy=(tp + tn) / len(gold), balanced_accuracy=(recall + specificity) / 2,
                spam_precision=precision, spam_recall=recall,
                spam_f1=2 * precision * recall / (precision + recall) if precision + recall else 0.,
                tp=tp, tn=tn, fp=fp, fn=fn, n=len(gold))


def prepare():
    out = HERE / 'data'
    if (out / 'manifest.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(URL, timeout=60).read()
    (out / 'source.zip').write_bytes(raw)
    archive = zipfile.ZipFile(io.BytesIO(raw))
    source = next(n for n in archive.namelist() if Path(n).name == 'SMSSpamCollection')
    rows, seen = [], set()
    raw_count = 0
    for line in archive.read(source).decode('utf-8-sig').splitlines():
        if not line.strip():
            continue
        label, text = line.split('\t', 1)
        assert label in LABELS
        raw_count += 1
        key = ' '.join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(id=hashlib.sha256(key.encode()).hexdigest(), text=text, label=LABELS.index(label), group=template(text)))
    groups = {}
    for row in rows:
        groups.setdefault(row['group'], []).append(row)
    conflicting = {k for k, v in groups.items() if len({r['label'] for r in v}) > 1}
    pools = {name: [] for name in ['train', 'validation', 'test']}
    rng = random.Random(20261026)
    for label in [0, 1]:
        keys = sorted(k for k, v in groups.items() if k not in conflicting and v[0]['label'] == label)
        rng.shuffle(keys)
        a, b = int(.70 * len(keys)), int(.85 * len(keys))
        for name, part in zip(pools, [keys[:a], keys[a:b], keys[b:]]):
            for key in part:
                pools[name].extend(groups[key])
    # Keep a small, reproducible training set; preserve each pool's class proportions.
    sizes = dict(train=1200, validation=400, test=800)
    splits = {}
    for name, pool in pools.items():
        size = min(sizes[name], len(pool))
        spam_n = round(size * sum(r['label'] for r in pool) / len(pool))
        selected = []
        for label, n in [(0, size-spam_n), (1, spam_n)]:
            candidates = sorted((r for r in pool if r['label'] == label), key=lambda r: r['id'])
            rng.shuffle(candidates)
            selected += candidates[:n]
        rng.shuffle(selected)
        splits[name] = selected
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in selected))
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not ({r['group'] for r in splits[a]} & {r['group'] for r in splits[b]})
    manifest = dict(source=URL, source_sha256=hashlib.sha256(raw).hexdigest(), raw_rows=raw_count,
                    unique_case_whitespace_messages=len(rows), conflicting_templates_removed=len(conflicting),
                    split_seed=20261026, grouping='Lowercase, replace URLs and digit runs, normalize punctuation/whitespace. Entire groups split before subsampling.',
                    splits={k: dict(n=len(v), spam=sum(r['label'] for r in v), ham=sum(r['label']==0 for r in v), sha256=digest(out/f'{k}.jsonl')) for k,v in splits.items()})
    write(out/'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


def rows(split):
    return [json.loads(line) for line in (HERE/'data'/f'{split}.jsonl').read_text().splitlines()]


def train(args):
    import torch
    from torch.nn import functional as F
    from transformers import GPT2Tokenizer
    from mingpt.model import GPT
    from mingpt.measurement import ChangeAudit, parameter_counts, cuda_memory
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    label_ids = [tokenizer.encode(' '+label) for label in LABELS]
    assert all(len(x) == 1 for x in label_ids), label_ids
    label_tokens = torch.tensor([x[0] for x in label_ids], device='cuda')
    prefix = tokenizer.encode('Classify this SMS as ham or spam.\nSMS: ')
    suffix = tokenizer.encode('\nLabel:')
    assert len(prefix) + 96 + len(suffix) <= 128
    def encode(data):
        result=[]
        for row in data:
            text_ids=tokenizer.encode(row['text'])
            ids=prefix+text_ids[:96]+suffix
            result.append((ids, row['label'], len(text_ids)>96))
        return result
    sets = {name: encode(rows(name)) for name in ['train', 'validation']}
    model = GPT.from_pretrained('gpt2').float().eval()
    base_hash = hashlib.sha256()
    for name, p in model.named_parameters():
        base_hash.update(name.encode()); base_hash.update(p.detach().numpy().tobytes())
    if args.method == 'lora':
        probe = torch.tensor(sets['train'][0][0]).unsqueeze(0)
        with torch.no_grad(): before=model(probe)[0]
        model.enable_lora(rank=4, alpha=8)
        with torch.no_grad(): torch.testing.assert_close(model(probe)[0], before, rtol=0, atol=0)
        del probe,before
    counts = parameter_counts(model)
    audit = ChangeAudit(model)
    model.to('cuda')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.)
    def batch(data):
        width=max(len(r[0]) for r in data)
        x=torch.full((len(data),width),tokenizer.eos_token_id, dtype=torch.long,device='cuda')
        for i,(ids,_,_) in enumerate(data): x[i,:len(ids)]=torch.tensor(ids,device='cuda')
        ends=torch.tensor([len(r[0])-1 for r in data],device='cuda')
        gold=torch.tensor([r[1] for r in data],device='cuda')
        return x,ends,gold
    def logits_for(data):
        x,ends,gold=batch(data)
        logits=model(x)[0][torch.arange(len(data),device='cuda'),ends]
        return logits,gold
    @torch.no_grad()
    def evaluate(data, predictions=False):
        model.eval(); total=0.; golds=[]; preds=[]; outputs=[]
        for start in range(0,len(data),8):
            logits,gold=logits_for(data[start:start+8])
            total+=F.cross_entropy(logits,label_tokens[gold],reduction='sum').item()
            scores=logits[:,label_tokens]
            pred=scores.argmax(-1)
            golds+=gold.tolist(); preds+=pred.tolist()
            outputs += scores.softmax(-1)[:,1].tolist()
        result=dict(label_nll=total/len(data), **metrics(golds,preds))
        if predictions: result['predictions']=[dict(id=row['id'],gold=int(g),pred=int(p),spam_probability=float(q)) for row,g,p,q in zip(rows('test'),golds,preds,outputs)]
        return result
    config={k:v for k,v in vars(args).items()}
    report=dict(config=config,task='SMS spam classification',torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),dtype='float32',tf32=False,parameters=counts,
                initial_base_sha256=base_hash.hexdigest(),data_manifest=json.loads((HERE/'data/manifest.json').read_text()),
                source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'mingpt/model.py',ROOT/'mingpt/measurement.py']},
                truncation_counts={k:sum(r[2] for r in data) for k,data in sets.items()},history=[],
                stopping_rule='First complete epoch with full training-set label NLL <= 0.05; otherwise cap at 5 epochs.',
                timing_scope='Compute: synchronized forward/backward/clip/optimizer, excluding transfers/evaluation/audits/I-O. Wall: training loop including validation, loss checks, and checkpoint writes, excluding model setup and final test/audit.')
    def save(name):
        weights={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        torch.save(dict(method=args.method,weights=weights,rank=4,alpha=8),out/f'{name}.pt')
    rng=random.Random(args.seed)
    order_hash=hashlib.sha256()
    elapsed=0.; first_stop=None; max_alloc=0; max_reserved=0
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    start_wall=time.perf_counter()
    for epoch in range(1,6):
        indices=list(range(len(sets['train']))); rng.shuffle(indices)
        order_hash.update(json.dumps(indices).encode())
        for start in range(0,len(indices),8):
            data=[sets['train'][i] for i in indices[start:start+8]]
            x,ends,gold=batch(data)
            model.train()
            torch.cuda.synchronize(); tick=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)[0][torch.arange(len(data),device='cuda'),ends]
            loss=F.cross_entropy(logits,label_tokens[gold])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize()
            elapsed+=time.perf_counter()-tick
            del logits,loss,x,ends,gold
        memory=cuda_memory('cuda')
        max_alloc=max(max_alloc,memory['peak_allocated_bytes']); max_reserved=max(max_reserved,memory['peak_reserved_bytes'])
        train_eval=evaluate(sets['train']); validation=evaluate(sets['validation'])
        entry=dict(epoch=epoch,steps=epoch*math.ceil(len(sets['train'])/8),train=train_eval,validation=validation,
                   training_step_seconds=elapsed,training_wall_seconds=time.perf_counter()-start_wall,
                   sample_order_sha256=order_hash.hexdigest(),peak_allocated_bytes=max_alloc,peak_reserved_bytes=max_reserved)
        report['history'].append(entry)
        if first_stop is None and train_eval['label_nll']<=.05:
            first_stop=epoch
            save('loss_stop')
        if epoch==5: save('epochs')
        entry['training_wall_seconds']=time.perf_counter()-start_wall
        print(json.dumps(dict(method=args.method,seed=args.seed,lr=args.lr,**entry)),flush=True)
        write(out/'progress.json',report)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    report['loss_stop_epoch']=first_stop or 5
    report['loss_target_reached']=first_stop is not None
    report['total_training_wall_seconds']=time.perf_counter()-start_wall
    changes=audit.measure(model)
    write(out/'changes.json',changes)
    report['final_changes']={k:v for k,v in changes.items() if k!='per_tensor'}
    write(out/'training.json',report)


def test(args):
    import torch
    from torch.nn import functional as F
    from transformers import GPT2Tokenizer
    from mingpt.model import GPT
    out=Path(args.out)
    report=json.loads((out/'training.json').read_text())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2',local_files_only=True)
    model=GPT.from_pretrained('gpt2').float().eval()
    if report['config']['method']=='lora': model.enable_lora(4,8)
    model.to('cuda')
    label_tokens=torch.tensor([tokenizer.encode(' '+label)[0] for label in LABELS],device='cuda')
    prefix=tokenizer.encode('Classify this SMS as ham or spam.\nSMS: '); suffix=tokenizer.encode('\nLabel:')
    data=rows('test')
    @torch.no_grad()
    def evaluate():
        golds=[];preds=[];outputs=[];total=0.;truncated=0
        for start in range(0,len(data),8):
            chunk=data[start:start+8]
            content=[tokenizer.encode(row['text']) for row in chunk]
            truncated+=sum(len(ids)>96 for ids in content)
            ids=[prefix+t[:96]+suffix for t in content]
            x=torch.full((len(ids),max(map(len,ids))),tokenizer.eos_token_id,dtype=torch.long,device='cuda')
            for i,t in enumerate(ids): x[i,:len(t)]=torch.tensor(t,device='cuda')
            ends=torch.tensor([len(t)-1 for t in ids],device='cuda')
            logits=model(x)[0][torch.arange(len(ids),device='cuda'),ends]
            gold=torch.tensor([r['label'] for r in chunk],device='cuda')
            total+=F.cross_entropy(logits,label_tokens[gold],reduction='sum').item()
            scores=logits[:,label_tokens]; pred=scores.argmax(-1); probs=scores.softmax(-1)[:,1]
            golds+=gold.tolist(); preds+=pred.tolist()
            outputs += [dict(id=r['id'],gold=int(g),pred=int(p),spam_probability=float(q)) for r,g,p,q in zip(chunk,gold.tolist(),pred.tolist(),probs.tolist())]
        return dict(**metrics(golds,preds),label_nll=total/len(data),truncated=truncated,predictions=outputs)
    results={}
    if args.base:
        if report['config']['method']=='lora':
            for module in model.modules():
                if hasattr(module,'scale'): module.scale=0.
        results['pretrained']=evaluate()
        if report['config']['method']=='lora':
            for module in model.modules():
                if hasattr(module,'scale'): module.scale=module.alpha/module.rank
    for rule in ['epochs','loss_stop']:
        checkpoint=out/f'{rule}.pt'
        if not checkpoint.exists(): checkpoint=out/'epochs.pt'
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
        missing,unexpected=model.load_state_dict(ckpt['weights'],strict=False)
        expected={n for n,p in model.named_parameters() if p.requires_grad}
        assert set(ckpt['weights'])==expected and not unexpected
        assert not (set(missing)&expected)
        model.eval()
        results[rule]=evaluate()
        del ckpt
    write(out/'test.json',results)
    print(json.dumps({k:{x:v for x,v in r.items() if x!='predictions'} for k,r in results.items()}),flush=True)


def run(args):
    prepare()
    root=HERE/'runs'
    if root.exists(): raise FileExistsError('Preserve existing runs; choose a fresh experiment directory')
    root.mkdir()
    protocol=dict(task='Real SMS spam/ham classification',seeds=[42,43,44],epochs=5,loss_target=.05,
                  train_size=1200,batch_size=8,max_message_tokens=96,rank=4,alpha=8,
                  learning_rates=[.001,.0001],selection='For each method, select LR by final epoch-5 validation label NLL on seed 42. No test access until choices are written.',
                  comparison='Same trajectory for fixed epochs and first loss-threshold checkpoint; shared loss checks allow direct checkpoint comparison. Wall times include loss checks for both rules.',
                  scoring='Higher next-token logit for the single-token labels ham/spam, not free-text answer matching. Default binary threshold only; no threshold tuning.',
                  limitations='One real dataset, three training seeds, one fixed data split, capped training subset, heuristic duplicate-template grouping; not a production filter or general theorem about LoRA.')
    write(HERE/'protocol.json',protocol)
    def call(mode,out,**kwargs):
        cmd=[sys.executable,str(Path(__file__).resolve()),mode,'--out',str(out)]
        for k,v in kwargs.items():
            cmd+=['--'+k.replace('_','-')]
            if v is not True: cmd += [str(v)]
        subprocess.run(cmd,check=True,cwd=ROOT)
    selected={}; candidates=[]
    for method in ['lora','full']:
        choices=[]
        for lr in [.001,.0001]:
            out=root/f'{method}-lr{lr}-seed42'
            call('train',out,method=method,lr=lr,seed=42)
            report=json.loads((out/'training.json').read_text())
            entry=dict(method=method,lr=lr,path=str(out),validation_nll=report['history'][-1]['validation']['label_nll'])
            choices.append(entry);candidates.append(entry)
        selected[method]=min(choices,key=lambda r:r['validation_nll'])
    write(HERE/'selection.json',dict(candidates=candidates,selected=selected,test_evaluated=False))
    for method in ['lora','full']:
        for seed in [43,44]:
            call('train',root/f'{method}-selected-seed{seed}',method=method,lr=selected[method]['lr'],seed=seed)
    # No further tuning occurs after this point.
    manifest=[]
    for method in ['lora','full']:
        for seed in [42,43,44]:
            out=Path(selected[method]['path']) if seed==42 else root/f'{method}-selected-seed{seed}'
            call('test',out,**({'base':True} if method=='lora' and seed==42 else {}))
            manifest.append(dict(method=method,seed=seed,path=str(out)))
    write(HERE/'completed.json',dict(runs=manifest,complete=True))
    print('ALL RUNS COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','train','test','run'])
    p.add_argument('--out');p.add_argument('--method',choices=['lora','full'],default='lora')
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--base',action='store_true')
    args=p.parse_args()
    if args.mode=='prepare': prepare()
    elif args.mode=='train': train(args)
    elif args.mode=='test': test(args)
    else: run(args)

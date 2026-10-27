"""FP32 LoRA adaptation to unseen match questions, with validation-only selection."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from mingpt.measurement import ChangeAudit, parameter_counts, cuda_memory
from lora_experiment import prompt, encode, load_rows, scales
from experiments.sports.scoring import score, aggregate


def write(path, data):
    path.write_text(json.dumps(data,indent=2))


@torch.no_grad()
def nll(model, data, device, batch_size):
    model.eval()
    total, tokens = 0., 0
    for start in range(0,len(data),batch_size):
        batch=data[start:start+batch_size]
        x,y=(torch.stack([r[i] for r in batch]).to(device) for i in range(2))
        n=int((y!=-1).sum())
        total+=model(x,y)[1].item()*n
        tokens+=n
    return total/tokens


@torch.no_grad()
def generate_batch(model, tokenizer, questions, device, new_tokens):
    """Right padding is strictly after each real prefix; causal attention hides it."""
    model.eval()
    ids=[tokenizer.encode(prompt(q)) for q in questions]
    starts=[len(x) for x in ids]
    width=max(starts)+new_tokens
    if width>model.block_size:
        raise ValueError('Prompt exceeds model context')
    x=torch.full((len(ids),width),tokenizer.eos_token_id,dtype=torch.long,device=device)
    for i,row in enumerate(ids):
        x[i,:len(row)]=torch.tensor(row,device=device)
    lengths=torch.tensor(starts,device=device)
    alive=torch.ones(len(ids),dtype=torch.bool,device=device)
    indices=torch.arange(len(ids),device=device)
    for _ in range(new_tokens):
        logits=model(x[:,:int(lengths.max())])[0]
        nxt=logits[indices,lengths-1].argmax(-1)
        x[indices[alive],lengths[alive]]=nxt[alive]
        lengths+=alive.long()
        alive &= nxt!=tokenizer.eos_token_id
        if not bool(alive.any()):
            break
    return [tokenizer.decode(x[i,start:int(lengths[i])].tolist(),skip_special_tokens=True).split('\nUser:')[0].strip() for i,start in enumerate(starts)]


def evaluate_outputs(model, tokenizer, rows, a):
    outputs=[]
    for start in range(0,len(rows),a.batch_size):
        batch=rows[start:start+a.batch_size]
        texts=generate_batch(model,tokenizer,[r['question'] for r in batch],a.device,a.new_tokens)
        for row,text in zip(batch,texts):
            outputs.append(dict(match_id=row['match_id'],kind=row['kind'],question=row['question'],
                                reference=row['answer'],output=text,scores=score(row,text)))
    metrics=aggregate(outputs)
    metrics['by_kind']={kind:aggregate([r for r in outputs if r['kind']==kind]) for kind in ['winner','halftime','comeback']}
    return dict(metrics=metrics,outputs=outputs)


def adapter(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def load_parameters(model, checkpoint):
    weights = checkpoint.get('weights', checkpoint.get('adapter'))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    expected_missing = {n for n, _ in model.state_dict().items()} - set(weights)
    if unexpected or set(missing) != expected_missing:
        raise RuntimeError('Checkpoint keys do not match the model')
    expected_weights = {n for n,p in model.named_parameters() if p.requires_grad}
    if set(weights) != expected_weights:
        raise RuntimeError('Checkpoint is missing trainable weights or contains unexpected weights')


def test_selected(model, tokenizer, a, report):
    root=Path('experiments/sports')
    # The test outputs are evaluated only after the validation-selected checkpoint is fixed.
    test_rows=load_rows(root/'test.jsonl')
    cf_rows=load_rows(root/'counterfactual.jsonl')
    test=encode(test_rows,tokenizer,a.length)
    training_answers={r['answer'] for r in load_rows(root/'train.jsonl')}
    report['reference_answers_seen_in_training']=sum(r['answer'] in training_answers for r in test_rows)
    for enabled,name in ([(True,'full')] if a.full else [(False,'base'),(True,'lora')]):
        scales(model,enabled)
        result=evaluate_outputs(model,tokenizer,test_rows,a)
        counter=evaluate_outputs(model,tokenizer,cf_rows,a)
        result['test_nll']=nll(model,test,a.device,a.batch_size)
        result['counterfactual_metrics']=counter['metrics']
        result['counterfactual_outputs']=counter['outputs']
        result['paired_primary_accuracy']=sum(x['scores']['primary_correct'] and y['scores']['primary_correct'] for x,y in zip(result['outputs'],counter['outputs']))/len(test_rows)
        result['paired_fields_accuracy']=sum(x['scores']['fields_correct'] and y['scores']['fields_correct'] for x,y in zip(result['outputs'],counter['outputs']))/len(test_rows)
        result['outputs_exactly_seen_in_training']=sum(r['output'] in training_answers for r in result['outputs'])
        report[name]=result
        print(name,json.dumps(result['metrics']),flush=True)
    scales(model,True)
    # Check batched decoding against independent single-example decoding on a fixed subset.
    for row,stored in zip(test_rows[:3],report['full' if a.full else 'lora']['outputs'][:3]):
        single=generate_batch(model,tokenizer,[row['question']],a.device,a.new_tokens)[0]
        if single!=stored['output']:
            raise RuntimeError('Batched and single-example greedy outputs disagree')
    report['batch_generation_checked']=True
    report['test_evaluated_after_validation_selection']=True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--epochs',type=int,help='Complete shuffled training passes; overrides --steps')
    p.add_argument('--target-train-loss',type=float,help='Stop when full training answer NLL reaches this threshold')
    p.add_argument('--checkpoint-selection',choices=['validation','final'],default='validation')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--length',type=int,default=128)
    p.add_argument('--rank',type=int,default=4)
    p.add_argument('--alpha',type=float,default=8)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--eval-every',type=int,default=300)
    p.add_argument('--audit-every',type=int,default=300)
    p.add_argument('--new-tokens',type=int,default=40)
    p.add_argument('--device',default='cuda')
    p.add_argument('--full',action='store_true',help='Full fine-tuning, with validation selection and held-out evaluation')
    p.add_argument('--skip-test',action='store_true',help='Validation-only learning-rate candidate')
    p.add_argument('--evaluate-only',action='store_true',help='Evaluate an already selected saved checkpoint')
    a=p.parse_args()
    if min(a.steps,a.batch_size,a.length,a.eval_every,a.audit_every,a.new_tokens)<1:
        p.error('Sizes must be positive')
    if a.epochs is not None and a.epochs < 1:
        p.error('Epochs must be positive')
    if a.target_train_loss is not None and (not math.isfinite(a.target_train_loss) or a.target_train_loss <= 0):
        p.error('Target training loss must be finite and positive')
    out=Path(a.out)
    if a.evaluate_only:
        report=json.loads((out/'report.json').read_text())
        device=a.device
        a=argparse.Namespace(**report['config'])
        a.device=device
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
        model=GPT.from_pretrained('gpt2').float()
        if not a.full: model.enable_lora(a.rank,a.alpha)
        checkpoint=torch.load(out/('model.pt' if a.full else 'adapter.pt'),map_location='cpu',weights_only=True)
        load_parameters(model,checkpoint)
        model.to(a.device).eval()
        if not (out/'training-report.json').exists(): write(out/'training-report.json',report)
        test_selected(model,tokenizer,a,report)
        write(out/'report.json',report)
        return
    if out.exists():
        raise FileExistsError(f'{out} already exists; preserve prior runs')
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
    root=Path('experiments/sports')
    train_rows=load_rows(root/'train.jsonl')
    val_rows=load_rows(root/'validation.jsonl')
    train=encode(train_rows,tokenizer,a.length)
    val=encode(val_rows,tokenizer,a.length)
    if a.epochs is not None:
        a.steps=a.epochs*math.ceil(len(train)/a.batch_size)
    model=GPT.from_pretrained('gpt2').float().eval()
    base_counts=parameter_counts(model)
    base_hash=hashlib.sha256()
    for name,param in model.named_parameters():
        base_hash.update(name.encode())
        base_hash.update(param.detach().numpy().tobytes())
    probe=train[0][0][:48].unsqueeze(0)
    with torch.no_grad():
        base_logits=model(probe)[0]
    if not a.full:
        model.enable_lora(a.rank,a.alpha)
    with torch.no_grad():
        torch.testing.assert_close(model(probe)[0],base_logits,rtol=0,atol=0)
    del base_logits,probe
    assert all(p.dtype==torch.float32 for p in model.parameters())
    audit=ChangeAudit(model)
    model.to(a.device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=0.)
    report=dict(config=vars(a),torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,
                initial_base_sha256=base_hash.hexdigest(),dtype='float32',tf32=False,parameters=parameter_counts(model),base_parameters=base_counts,
                initial_model_memory=cuda_memory(a.device),
                data_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.jsonl')},
                source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path('sports_experiment.py'),Path('lora_experiment.py'),Path('mingpt/model.py'),Path('mingpt/measurement.py'),root/'data.py',root/'scoring.py']},
                history=[],validation_history=[],memory_windows=[])
    best=float('inf')
    best_step=None
    step_times=[]
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    if a.device=='cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    sample_hash=hashlib.sha256()
    supervised_tokens=0
    start_wall=time.perf_counter()
    epoch_order=[]
    epoch_offset=0
    examples_processed=0
    stopped_for_loss=False
    for step in range(1,a.steps+1):
        if a.epochs is None:
            indices=random.choices(range(len(train)),k=a.batch_size)
        else:
            if epoch_offset >= len(epoch_order):
                epoch_order=list(range(len(train)))
                random.shuffle(epoch_order)
                epoch_offset=0
            indices=epoch_order[epoch_offset:epoch_offset+a.batch_size]
            epoch_offset+=len(indices)
        examples_processed+=len(indices)
        sample_hash.update(json.dumps(indices).encode())
        batch=[train[i] for i in indices]
        x,y=(torch.stack([r[i] for r in batch]).to(a.device) for i in range(2))
        supervised_tokens+=int((y!=-1).sum())
        model.train()
        if a.device=='cuda': torch.cuda.synchronize()
        start=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss=model(x,y)[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        if a.device=='cuda': torch.cuda.synchronize()
        step_times.append(time.perf_counter()-start)
        if step in (1,40):
            report['gradient_bytes']=sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None)
            report[f'training_memory_through_step_{step}']=cuda_memory(a.device)
        evaluate_now=step%a.eval_every==0 or step==a.steps or (a.epochs is not None and epoch_offset>=len(train))
        if evaluate_now:
            # Capture training memory before loss evaluation creates other temporaries.
            report['memory_windows'].append(cuda_memory(a.device))
        if evaluate_now and a.target_train_loss is not None:
            train_nll=nll(model,train,a.device,a.batch_size)
            stopped_for_loss=train_nll<=a.target_train_loss
            report.setdefault('training_nll_history',[]).append(dict(step=step,nll=train_nll))
            print(f'training NLL step={step} nll={train_nll:.6f}',flush=True)
        if step==1 or step%a.audit_every==0 or evaluate_now:
            changes=audit.measure(model)
            write(out/f'changes-{step:04d}.json',changes)
            entry=dict(step=step,train_loss=loss.item(),changed_elements=changes['changed_elements'],frozen_changed_elements=changes['frozen_changed_elements'])
            report['history'].append(entry)
            print(json.dumps(entry),flush=True)
        if evaluate_now:
            validation=nll(model,val,a.device,a.batch_size)
            report['validation_history'].append(dict(step=step,nll=validation))
            if report['history'][-1]['step']==step:
                report['history'][-1]['validation_nll']=validation
            print(f'validation step={step} nll={validation:.6f}',flush=True)
            if validation<best or (a.checkpoint_selection=='final' and (step==a.steps or stopped_for_loss)):
                best,best_step=validation,step
                torch.save(dict(base='gpt2',full=a.full,rank=a.rank,alpha=a.alpha,step=step,weights=adapter(model)),out/('model.pt' if a.full else 'adapter.pt'))
            write(out/'progress.json',report)
            optimizer.zero_grad(set_to_none=True)
            if a.device=='cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        if stopped_for_loss:
            break
    report['completed_steps']=step
    report['examples_processed']=examples_processed
    report['completed_epochs']=examples_processed/len(train)
    report['stop_reason']='target_train_loss' if stopped_for_loss else ('epoch_limit' if a.epochs is not None else 'step_limit')
    report['final_training_nll']=nll(model,train,a.device,a.batch_size)
    report['sample_order_sha256']=sample_hash.hexdigest()
    report['supervised_training_tokens']=supervised_tokens
    report['processed_training_tokens']=examples_processed*a.length
    report['training_wall_seconds']=time.perf_counter()-start_wall
    report['training_step_seconds']=sum(step_times)
    report['step_seconds']=step_times
    report['warm_training_steps_per_second']=(len(step_times)-10)/sum(step_times[10:]) if len(step_times)>10 else None
    report['timing_scope']='Synchronized forward/backward/clipping/optimizer. Excludes data transfer, audits and validation. Warm throughput excludes first 10 steps.'
    report['training_peak_allocated_bytes']=max(m['peak_allocated_bytes'] for m in report['memory_windows']) if a.device=='cuda' else None
    report['training_peak_reserved_bytes']=max(m['peak_reserved_bytes'] for m in report['memory_windows']) if a.device=='cuda' else None
    report['optimizer_tensor_bytes']=sum(v.numel()*v.element_size() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
    report['final_changes']={k:v for k,v in changes.items() if k!='per_tensor'}
    report['selected_step']=best_step
    report['selected_validation_nll']=best
    checkpoint=torch.load(out/('model.pt' if a.full else 'adapter.pt'),map_location='cpu',weights_only=True)
    load_parameters(model,checkpoint)
    selected_changes=audit.measure(model)
    write(out/'selected-changes.json',selected_changes)
    report['selected_changes']={k:v for k,v in selected_changes.items() if k!='per_tensor'}
    if not a.skip_test:
        test_selected(model,tokenizer,a,report)
    write(out/'report.json',report)
    print(f'Saved {out}/report.json',flush=True)


if __name__=='__main__':
    main()

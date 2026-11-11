"""Shared GPU operations. Attention/cache knobs never change retained context."""
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import torch
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from lora_experiment import encode,load_rows
from experiments.paper_hyperparameters.common import ROOT,read,sha
from experiments.paper_hyperparameters.protocol import DESIGN,BASE_TILE,BASE_CHUNK
from experiments.paper_hyperparameters import fa_kernel
import mingpt.tiled_attention as tiled

DATA=ROOT/'experiments/paper_hyperparameters/data'


def amp():return torch.autocast('cuda',dtype=torch.float16)


def tokenizer(study=None):
    tok=GPT2Tokenizer.from_pretrained('gpt2',local_files_only=True)
    if study and (study/'preflight.json').exists():
        digest=hashlib.sha256(json.dumps(tok.get_vocab(),sort_keys=True).encode()).hexdigest()
        assert digest==read(study/'preflight.json')['tokenizer_sha256'],'Tokenizer drift'
    return tok


def encoded(split,tok):return encode(load_rows(DATA/(split+'.jsonl')),tok,DESIGN['train']['length'])


def fingerprint(model,frozen=True):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if frozen and p.requires_grad:continue
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_model(config,seed,checkpoint=None,study=None):
    torch.manual_seed(seed)
    model=GPT.from_pretrained('gpt2').float()
    model.enable_lora(config['rank'],config['alpha'])
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    if study and (study/'preflight.json').exists():
        assert fingerprint(model)==read(study/'preflight.json')['frozen_base_sha256'],'Pretrained weights changed'
    if checkpoint:
        report=read(Path(checkpoint).parent/'training.json')
        assert report['complete'] and sha(checkpoint)==report['adapter_sha256'],'Adapter checkpoint changed'
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert set(weights)=={n for n,p in model.named_parameters() if p.requires_grad}
        missing,unexpected=model.load_state_dict(weights,strict=False)
        assert not unexpected and all(not n.endswith(('A.weight','B.weight')) for n in missing)
    return model.cuda().eval()


@contextmanager
def backend(model,name='eager',tiles=BASE_TILE):
    original=tiled.tiled_attention_fp32
    fa_kernel.TILES=tuple(tiles)
    tiled.tiled_attention_fp32=fa_kernel.attention
    model.set_attention_backend('tiled_fp32' if name=='prototype' else name)
    try:yield
    finally:
        tiled.tiled_attention_fp32=original;model.set_attention_backend('eager')


@torch.inference_mode()
def likelihood(model,data,setting=None):
    setting=setting or dict(backend='eager',cached=False,chunk=BASE_CHUNK,tiles=BASE_TILE)
    model.eval();total=count=0
    with backend(model,setting['backend'],setting['tiles']),amp():
        for i in range(0,len(data),8):
            x,y=(torch.stack([r[k] for r in data[i:i+8]]).cuda() for k in (0,1))
            if setting['cached']:
                cache=None
                for offset in range(0,x.size(1),setting['chunk']):
                    logits,_,cache=model(x[:,offset:offset+setting['chunk']],past_key_values=cache,use_cache=True)
                    targets=y[:,offset:offset+setting['chunk']]
                    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),ignore_index=-1,reduction='sum')
                    total+=float(loss);count+=int((targets!=-1).sum())
            else:
                logits,_=model(x)
                loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=-1,reduction='sum')
                total+=float(loss);count+=int((y!=-1).sum())
    return total/count


def setting(cached=False,chunk=BASE_CHUNK,attention='eager',tiles=BASE_TILE):
    return dict(cached=cached,chunk=chunk,backend=attention,tiles=list(tiles))


@torch.inference_mode()
def generate(model,ids,new_tokens,cached=False,chunk=BASE_CHUNK):
    if new_tokens<1 or chunk<1 or ids.size(1)+new_tokens>model.block_size:raise ValueError('Invalid fixed workload/context')
    cache=None;output=ids
    if cached:
        for start in range(0,ids.size(1),chunk):
            logits,_,cache=model(ids[:,start:start+chunk],past_key_values=cache,use_cache=True,last_token_only=True)
    for i in range(new_tokens):
        if not cached:logits,_=model(output,last_token_only=True)
        elif i:logits,_,cache=model(output[:,-1:],past_key_values=cache,use_cache=True,last_token_only=True)
        output=torch.cat((output,logits[:,-1].argmax(-1,keepdim=True)),dim=1)
    return output


@torch.inference_mode()
def timed(model,ids,new_tokens,config):
    model.eval()
    # Backend dispatch, GC, host/device input transfer and telemetry excluded equally.
    with backend(model,config['backend'],config['tiles']),amp():
        start,stop=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
        begin=time.perf_counter();start.record()
        output=generate(model,ids,new_tokens,config['cached'],config['chunk'])
        stop.record();torch.cuda.synchronize()
        result=dict(wall_ms=(time.perf_counter()-begin)*1000,cuda_ms=start.elapsed_time(stop),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    baseline_allocated_bytes=base,incremental_peak_bytes=torch.cuda.max_memory_allocated()-base)
    result['output_sha256']=hashlib.sha256(output.cpu().numpy().tobytes()).hexdigest()
    return result


def cleanup():gc.collect();torch.cuda.empty_cache()


@torch.inference_mode()
def kernel_checks(tiles):
    fa_kernel.TILES=tuple(tiles);rows=[]
    for nq,past in [(37,0),(17,21),(1,36),(129,0),(1,895),(1,1023)]:
        torch.manual_seed(700+nq+past)
        q=torch.randn(1,2,nq,64,device='cuda',dtype=torch.float16)
        k=torch.randn(1,2,nq+past,64,device='cuda',dtype=torch.float16);v=torch.randn_like(k)
        got=fa_kernel.attention(q,k,v,past)
        mask=torch.arange(nq+past,device='cuda')[None,:]<=past+torch.arange(nq,device='cuda')[:,None]
        reference=torch.softmax((q.double()@k.double().transpose(-2,-1)/8).masked_fill(~mask,-float('inf')),-1)@v.double()
        torch.testing.assert_close(got.double(),reference,atol=DESIGN['quality']['kernel_atol'],rtol=DESIGN['quality']['kernel_rtol'])
        rows.append(dict(nq=nq,past=past,max_error=float((got.double()-reference).abs().max())))
    return rows


@torch.inference_mode()
def cache_checks(model,chunk):
    rows=[]
    for b,p in [(1,37),(1,896),(4,129)]:
        ids=torch.randint(50257,(b,p+3),generator=torch.Generator().manual_seed(8800+b+p)).cuda()
        cache=None
        with amp():
            for i in range(0,p,chunk):
                last,_,cache=model(ids[:,i:min(i+chunk,p)],past_key_values=cache,use_cache=True,last_token_only=True)
            for step in range(4):
                if step:last,_,cache=model(ids[:,p+step-1:p+step],past_key_values=cache,use_cache=True,last_token_only=True)
                reference=model(ids[:,:p+step],last_token_only=True)[0]
                torch.testing.assert_close(last.float(),reference.float(),atol=DESIGN['quality']['cache_logit_atol'],rtol=DESIGN['quality']['cache_logit_rtol'])
                rows.append(dict(batch=b,prompt=p,step=step,max_error=float((last.float()-reference.float()).abs().max()),
                                 argmax_equal=bool(torch.equal(last[:,-1].argmax(-1),reference[:,-1].argmax(-1)))))
    return rows

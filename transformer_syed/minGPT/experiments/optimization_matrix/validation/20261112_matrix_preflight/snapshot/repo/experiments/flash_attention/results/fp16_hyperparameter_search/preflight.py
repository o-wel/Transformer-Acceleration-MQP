"""GPU gates: real updates, projection/operand/cache dtypes and numerical checks."""
import itertools
import gc


def run(s, out):
    torch=s.torch
    dtype=torch.float16 if s.PRECISION=='fp16' else torch.float32
    import attention
    import cache
    from experiments.kv_cache.benchmark import timed_generate
    checks=[]
    for bm,bn,w in itertools.product([16,32],[32,64],[4,8]):
        s.configure_attention('fa1',bm,bn,w)
        for nq,past in [(35,0),(1,35),(7,35),(1,895)]:
            q=torch.randn(1,2,nq,64,device='cuda',dtype=dtype,requires_grad=True)
            k=torch.randn(1,2,nq+past,64,device='cuda',dtype=dtype,requires_grad=True)
            v=torch.randn_like(k,requires_grad=True)
            mask=torch.arange(nq+past,device='cuda')[None,:] <= past+torch.arange(nq,device='cuda')[:,None]
            # FP64 reference distinguishes mathematical accuracy from FP16 rounding.
            ref=torch.softmax((q.double()@k.double().transpose(-2,-1))*.125+torch.where(mask,0.,-torch.inf),-1)@v.double()
            got=attention.attention(q,k,v,past)
            assert got.dtype==dtype
            tol=(.012,.04) if s.PRECISION=='fp16' else (3e-4,3e-4)
            torch.testing.assert_close(got.double(),ref,atol=tol[0],rtol=tol[1])
            grad=torch.randn_like(got)
            a=torch.autograd.grad(got,(q,k,v),grad,retain_graph=True)
            b=torch.autograd.grad(ref,(q,k,v),grad.double())
            for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=tol[0],rtol=tol[1])
            checks.append(dict(bm=bm,bn=bn,warps=w,nq=nq,past=past,operand_dtype=str(dtype),output_dtype=str(got.dtype),max_abs_error=float((got-ref).abs().max().detach())))
    # Both full fine-tuning and LoRA must complete actual scaled optimizer updates.
    trials=[]
    for method in ['lora','full']:
        trials.append(s.trial(out/'preflight_trials',dict(s.grid()[0],method=method),16))
    model=s.build(s.grid()[0],42).eval()
    caches=[]
    for mode in ['eager','fa1','hybrid']:
        s.configure_attention(mode)
        model.set_attention_backend('eager' if mode=='eager' else 'tiled_fp32')
        for static in [False,True]:
            cache.enable(model,static)
            x=torch.randint(0,50257,(2,39),device='cuda')
            with torch.inference_mode():
                reference=model(x)[0]
                past=None;pieces=[]
                for start in range(0,39,7):
                    logits,_,past=model(x[:,start:start+7],past_key_values=past,use_cache=True)
                    assert all(k.dtype==v.dtype==dtype for k,v in past)
                    pieces.append(logits)
                actual=torch.cat(pieces,dim=1)
                torch.testing.assert_close(actual,reference,atol=.1 if s.PRECISION=='fp16' else 3e-4,rtol=.02 if s.PRECISION=='fp16' else 3e-4)
                timed_generate(model,x,8,True)
                caches.append(dict(backend=mode,static=static,dtype=str(past[0][0].dtype),max_abs_error=float((actual-reference).abs().max())))
    s.write(out/'smoke.json',dict(complete=True,task=s.TASK,precision=s.PRECISION,gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,attention_checks=checks,cache_checks=caches,dtype_audit=model._dtype_audit,
        optimizer_updates=[dict(method=r['config']['method'],steps=r['steps'],audit=r['dtype_audit']) for r in trials]))
    del model
    gc.collect();torch.cuda.empty_cache()

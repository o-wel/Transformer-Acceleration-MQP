"""FA1-derived Algorithm 1/4 prototype: K/V tiles outer, query tiles inner.

Identical code/scheduling for FP32 and FP16. This is not the original FA1 CUDA
implementation, and makes no performance equivalence claim to that implementation.
One Triton program per batch/head owns output and gradient state, avoiding atomics.
Normalized O and row statistics are written between key tiles as in FA1; no
FA2 query-tile parallel grid or delayed-normalization forward is used.
"""
import torch,triton
import triton.language as tl

@triton.jit
def _fwd(Q,K,V,O,M,L,LSE,H:tl.constexpr,NQ,NK,D:tl.constexpr,PAST,BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr):
    bh=tl.program_id(0);di=tl.arange(0,BD);ri=tl.arange(0,BM);cj=tl.arange(0,BN)
    for j in range(tl.cdiv(NK,BN)):
        kj=j*BN+cj
        k=tl.load(K+(bh*NK+kj[:,None])*D+di[None,:],(kj[:,None]<NK)&(di[None,:]<D),0)
        v=tl.load(V+(bh*NK+kj[:,None])*D+di[None,:],(kj[:,None]<NK)&(di[None,:]<D),0)
        start=tl.maximum(0,j*BN-PAST)//BM*BM
        for i in range(start,NQ,BM):
            qi=i+ri
            q=tl.load(Q+(bh*NQ+qi[:,None])*D+di[None,:],(qi[:,None]<NQ)&(di[None,:]<D),0)
            oldm=tl.load(M+bh*NQ+qi,qi<NQ,-float('inf'))
            oldl=tl.load(L+bh*NQ+qi,qi<NQ,0)
            oldo=tl.load(O+(bh*NQ+qi[:,None])*D+di[None,:],(qi[:,None]<NQ)&(di[None,:]<D),0)
            s=tl.dot(q,tl.trans(k),input_precision='ieee')*(D**-0.5)
            mask=(kj[None,:]<NK)&(kj[None,:]<=PAST+qi[:,None])
            s=tl.where(mask,s,-float('inf'))
            m=tl.maximum(oldm,tl.max(s,1));m=tl.where(m==-float('inf'),0.,m)
            p=tl.exp(s-m[:,None]);correction=tl.where(oldl>0,tl.exp(oldm-m),0.)
            newl=oldl*correction+tl.sum(p,1)
            acc=oldo*(oldl*correction)[:,None]+tl.dot(p.to(v.dtype),v,input_precision='ieee')
            out=acc/tl.where(newl>0,newl,1.)[:,None]
            tl.store(O+(bh*NQ+qi[:,None])*D+di[None,:],out,(qi[:,None]<NQ)&(di[None,:]<D))
            tl.store(M+bh*NQ+qi,m,qi<NQ);tl.store(L+bh*NQ+qi,newl,qi<NQ)
            tl.store(LSE+bh*NQ+qi,m+tl.log(newl),qi<NQ)

@triton.jit
def _bwd(Q,K,V,DO,LSE,DELTA,DQ,DK,DV,NQ,NK,D:tl.constexpr,PAST,BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr):
    bh=tl.program_id(0);di=tl.arange(0,BD);ri=tl.arange(0,BM);cj=tl.arange(0,BN)
    for j in range(tl.cdiv(NK,BN)):
        kj=j*BN+cj
        k=tl.load(K+(bh*NK+kj[:,None])*D+di[None,:],(kj[:,None]<NK)&(di[None,:]<D),0)
        v=tl.load(V+(bh*NK+kj[:,None])*D+di[None,:],(kj[:,None]<NK)&(di[None,:]<D),0)
        dk=tl.zeros((BN,BD),tl.float32);dv=tl.zeros((BN,BD),tl.float32)
        start=tl.maximum(0,j*BN-PAST)//BM*BM
        for i in range(start,NQ,BM):
            qi=i+ri
            q=tl.load(Q+(bh*NQ+qi[:,None])*D+di[None,:],(qi[:,None]<NQ)&(di[None,:]<D),0)
            do=tl.load(DO+(bh*NQ+qi[:,None])*D+di[None,:],(qi[:,None]<NQ)&(di[None,:]<D),0)
            lse=tl.load(LSE+bh*NQ+qi,qi<NQ,0);delta=tl.load(DELTA+bh*NQ+qi,qi<NQ,0)
            s=tl.dot(q,tl.trans(k),input_precision='ieee')*(D**-0.5)
            visible=(qi[:,None]<NQ)&(kj[None,:]<NK)&(kj[None,:]<=PAST+qi[:,None])
            p=tl.exp(tl.where(visible,s-lse[:,None],-float('inf')))
            dp=tl.dot(do,tl.trans(v),input_precision='ieee')
            ds=p*(dp-delta[:,None])
            olddq=tl.load(DQ+(bh*NQ+qi[:,None])*D+di[None,:],(qi[:,None]<NQ)&(di[None,:]<D),0)
            dq=olddq+tl.dot(ds.to(k.dtype),k,input_precision='ieee')*(D**-0.5)
            dk+=tl.dot(tl.trans(ds.to(q.dtype)),q,input_precision='ieee')*(D**-0.5)
            dv+=tl.dot(tl.trans(p.to(do.dtype)),do,input_precision='ieee')
            tl.store(DQ+(bh*NQ+qi[:,None])*D+di[None,:],dq,(qi[:,None]<NQ)&(di[None,:]<D))
        tl.store(DK+(bh*NK+kj[:,None])*D+di[None,:],dk,(kj[:,None]<NK)&(di[None,:]<D))
        tl.store(DV+(bh*NK+kj[:,None])*D+di[None,:],dv,(kj[:,None]<NK)&(di[None,:]<D))

TILES = (16, 32, 4)

def forward(q,k,v,past):
    q,k,v=[x.contiguous() for x in (q,k,v)];b,h,nq,d=q.shape
    o=torch.zeros(q.shape,device=q.device,dtype=torch.float32)
    m=torch.full((b,h,nq),-float('inf'),device=q.device);l=torch.zeros_like(m);lse=torch.empty_like(m)
    kernel=_fwd[(b*h,)](q,k,v,o,m,l,lse,h,nq,k.size(2),d,past,TILES[0],TILES[1],max(16,triton.next_power_of_2(d)),num_warps=TILES[2])
    return o.to(q.dtype),(q,k,v,o,lse),kernel

class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,past):
        out,saved,kernel=forward(q,k,v,past);ctx.save_for_backward(*saved);ctx.past=past;ctx.tiles=TILES;return out
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,grad):
        q,k,v,o,lse=ctx.saved_tensors;grad=grad.contiguous();b,h,nq,d=q.shape
        delta=(o*grad.float()).sum(-1).contiguous()
        dq=torch.zeros(q.shape,device=q.device,dtype=torch.float32)
        dk=torch.empty(k.shape,device=k.device,dtype=torch.float32);dv=torch.empty_like(dk)
        _bwd[(b*h,)](q,k,v,grad,lse,delta,dq,dk,dv,nq,k.size(2),d,ctx.past,TILES[0],TILES[1],max(16,triton.next_power_of_2(d)),num_warps=TILES[2])
        return dq.to(q.dtype),dk.to(k.dtype),dv.to(v.dtype),None

def attention(q,k,v,past=0):
    assert q.dtype in (torch.float32,torch.float16) and q.dtype==k.dtype==v.dtype
    assert q.device.type=='cuda' and k.shape==v.shape and k.shape[:2]==q.shape[:2] and k.size(2)==past+q.size(2)
    return Attention.apply(q,k,v,past) if torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,v)) else forward(q,k,v,past)[0]

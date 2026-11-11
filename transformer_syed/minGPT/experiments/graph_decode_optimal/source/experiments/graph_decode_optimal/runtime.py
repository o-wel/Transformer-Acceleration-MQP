"""Graph-compatible exact-context inference with device-resident positions.

Preserves ordinary GPT-2 learned positions and FP16 serving arithmetic. The
attention algorithm is the earlier online-softmax prototype; graph replay is
an established optimization, not claimed as a new algorithm.
"""
import time
import torch
import triton
import triton.language as tl
from torch.nn import functional as F
from mingpt.model import LoRALinear
from experiments.joint_decode.kernel import _decode


@triton.jit
def _device_position_decode(QKV,Z,UP,KC,VC,O,POS,C:tl.constexpr,H:tl.constexpr,
        D:tl.constexpr,R:tl.constexpr,RR:tl.constexpr,CAP:tl.constexpr,
        SCALE:tl.constexpr,FUSED:tl.constexpr,BN:tl.constexpr):
    past=tl.load(POS).to(tl.int32)
    _decode(QKV,Z,UP,KC,VC,O,C,H,D,R,RR,CAP,past,SCALE,FUSED,BN)


def attention(qkv,cache,pos,heads,z=None,up=None,scale=1.,bn=128,warps=4):
    b,_,c3=qkv.shape;c=c3//3
    fused=z is not None;r=z.size(-1) if fused else 1
    out=torch.empty((b,1,c),device=qkv.device,dtype=qkv.dtype)
    _device_position_decode[(b,heads)](qkv,z if fused else qkv,up if fused else qkv,
        cache[0],cache[1],out,pos,c,heads,c//heads,r,triton.next_power_of_2(r),
        cache[0].size(2),scale,fused,bn,num_warps=warps,enable_fp_fusion=False)
    return out


class Runner:
    """One reusable graph per model/batch; each request resets its own cache.

    Only caller-owned buffer state changes. Changing adapter/model weights
    requires a new runner/capture; context rollover is explicitly unsupported.
    Construction/capture is reported separately from warm request latency.
    """
    @torch.inference_mode()
    def __init__(self,model,batch,fused,graph,bn=128,warps=4):
        self.bn,self.warps=bn,warps
        self.model,self.batch,self.fused,self.graphed=model,batch,fused,graph
        assert not model.training and next(model.parameters()).dtype==torch.float16
        self.capacity=model.block_size
        device=next(model.parameters()).device
        self.position=torch.tensor([16],device=device,dtype=torch.long)
        self.write_index=torch.zeros((1,1),device=device,dtype=torch.long)
        self.token=torch.full((batch,1),42,device=device,dtype=torch.long)
        self.output=torch.empty((batch,self.capacity),device=device,dtype=torch.long)
        self.caches=[]
        for block in model.transformer.h:
            h=block.attn.n_head;d=block.attn.n_embd//h
            self.caches.append(tuple(torch.zeros((batch,h,self.capacity,d),device=device,
                dtype=torch.float16) for _ in range(2)))
        torch.cuda.synchronize();start=time.perf_counter()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph=None
        if graph:
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.step()
            torch.cuda.synchronize()
        self.setup_ms=(time.perf_counter()-start)*1000

    def step(self):
        m=self.model
        x=m.transformer.wte(self.token)+m.transformer.wpe(self.position.view(1,1))
        for i,block in enumerate(m.transformer.h):
            a=block.attn;norm=block.ln_1(x)
            if self.fused and isinstance(a.c_attn,LoRALinear):
                qkv=a.c_attn.linear(norm);z=a.c_attn.A(norm)
                y=attention(qkv,self.caches[i],self.position,a.n_head,z,a.c_attn.B.weight,a.c_attn.scale,self.bn,self.warps)
            else:
                y=attention(a.c_attn(norm),self.caches[i],self.position,a.n_head,bn=self.bn,warps=self.warps)
            x=x+a.c_proj(y)
            x=x+block.mlpf(block.ln_2(x))
        self.step_logits=m.lm_head(m.transformer.ln_f(x))
        # Match the existing generate() softmax/topk policy, including FP16 ties.
        nxt=torch.topk(F.softmax(self.step_logits[:,-1,:],dim=-1),k=1,dim=-1)[1]
        self.output.scatter_(1,self.write_index.expand(self.batch,1),nxt)
        self.token.copy_(nxt)
        self.position.add_(1);self.write_index.add_(1)

    def replay(self):
        if self.graphed:self.graph.replay()
        else:self.step()

    @torch.inference_mode()
    def prefill(self,ids):
        assert ids.shape[0]==self.batch and 0<ids.size(1)<self.capacity
        logits,_,past=self.model(ids,use_cache=True,last_token_only=True)
        self.prefill_logits=logits
        for buf,pair in zip(self.caches,past):
            for target,source in zip(buf,pair):target[:,:,:ids.size(1)].copy_(source)
        nxt=torch.topk(F.softmax(logits[:,-1,:],dim=-1),k=1,dim=-1)[1]
        self.token.copy_(nxt);self.output[:,:1].copy_(nxt)
        self.position.fill_(ids.size(1));self.write_index.fill_(1)

    @torch.inference_mode()
    def generate(self,ids,count):
        assert count>0 and ids.size(1)+count<=self.capacity
        self.prefill(ids)
        for _ in range(count-1):self.replay()
        return torch.cat((ids,self.output[:,:count]),dim=1)

"""Inference-only preallocated KV storage ablation, using the existing cache API."""
import math
from types import MethodType
import torch


def forward(self, x, past_key_value=None, use_cache=False):
    if not use_cache:
        return self._dynamic_forward(x, past_key_value, use_cache)
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('Static KV cache is inference-only')
    b,t,c=x.shape
    q,k,v=[z.view(b,t,self.n_head,c//self.n_head).transpose(1,2)
           for z in self.c_attn(x).split(self.n_embd,dim=2)]
    past=0 if past_key_value is None else past_key_value[0].size(2)
    shape=(b,self.n_head,self.bias.size(-1),c//self.n_head)
    if not hasattr(self,'_storage') or self._storage[0].shape != shape or self._storage[0].dtype != k.dtype:
        self._storage=(torch.empty(shape,device=k.device,dtype=k.dtype),torch.empty(shape,device=k.device,dtype=v.dtype))
    if past+t>shape[2]:raise ValueError('Static cache capacity exceeded')
    # The previous view must belong to this cache; concurrent requests need
    # separate model/cache instances. This runner uses one serial request stream.
    if past and past_key_value[0].data_ptr()!=self._storage[0].data_ptr():
        raise ValueError('Static cache ownership mismatch')
    self._storage[0][:,:,past:past+t].copy_(k)
    self._storage[1][:,:,past:past+t].copy_(v)
    k,v=[z[:,:,:past+t] for z in self._storage]
    if self.attention_backend=='eager':
        scores=(q@k.transpose(-2,-1))/math.sqrt(k.size(-1))
        scores=scores.masked_fill(self.bias[:,:,past:past+t,:past+t]==0,-torch.inf)
        y=torch.softmax(scores,-1)@v
    elif self.attention_backend=='tiled_fp32':
        from mingpt.tiled_attention import tiled_attention_fp32
        y=tiled_attention_fp32(q,k,v,past)
    else:raise ValueError('Static study cache supports eager and the explicit custom kernel')
    y=y.transpose(1,2).contiguous().view(b,t,c)
    return self.resid_dropout(self.c_proj(y)),(k,v)


def enable(model, enabled):
    for block in model.transformer.h:
        attn=block.attn
        if enabled and not hasattr(attn,'_dynamic_forward'):
            attn._dynamic_forward=attn.forward
            attn.forward=MethodType(forward,attn)
        elif not enabled and hasattr(attn,'_dynamic_forward'):
            attn.forward=attn._dynamic_forward
            del attn._dynamic_forward
        if hasattr(attn,'_storage'):
            del attn._storage


def allocated_bytes(model):
    return sum(z.numel()*z.element_size() for block in model.transformer.h
               for z in getattr(block.attn,'_storage',()))

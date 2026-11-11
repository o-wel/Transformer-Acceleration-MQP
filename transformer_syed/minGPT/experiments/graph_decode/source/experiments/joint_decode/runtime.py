"""Isolated experimental runtime; production mingpt/model.py is untouched."""
from contextlib import contextmanager
from types import MethodType
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from mingpt.model import LoRALinear
from experiments.joint_decode.kernel import decode, expand


def _attention(self, x, past_key_value=None, use_cache=False):
    assert use_cache and not self.training and not torch.is_grad_enabled()
    batch, length, c = x.shape
    past = 0 if past_key_value is None else past_key_value[0].size(2)
    if past_key_value is None:
        self._buffers_kv = tuple(torch.empty((batch, self.n_head, self._capacity,
            c // self.n_head), device=x.device, dtype=x.dtype) for _ in range(2))
    kbuf, vbuf = self._buffers_kv
    assert past + length <= self._capacity
    fused = self._joint_mode in {'fused_stream', 'expand_stream'} and past_key_value is not None and length == 1
    stream = self._joint_mode in {'static_stream', 'fused_stream', 'expand_stream'} and past_key_value is not None and length == 1
    if fused:
        assert isinstance(self.c_attn, LoRALinear)
        qkv = self.c_attn.linear(x)
        z = self.c_attn.A(x)
        if self._joint_mode == 'expand_stream':
            qkv = expand(qkv,z,self.c_attn.B.weight,self.c_attn.scale)
            y = decode(qkv,(kbuf,vbuf),past,self.n_head)
        else:
            y = decode(qkv, (kbuf, vbuf), past, self.n_head,
                       z=z, up=self.c_attn.B.weight, scale=self.c_attn.scale)
    else:
        qkv = self.c_attn(x)
        if stream:
            y = decode(qkv, (kbuf, vbuf), past, self.n_head)
        else:
            q, k, v = [t.view(batch, length, self.n_head, c // self.n_head).transpose(1, 2)
                       for t in qkv.split(c, dim=2)]
            kbuf[:, :, past:past + length].copy_(k)
            vbuf[:, :, past:past + length].copy_(v)
            assert past == 0 or length == 1
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                y = F.scaled_dot_product_attention(q, kbuf[:, :, :past + length],
                    vbuf[:, :, :past + length], is_causal=past == 0 and length > 1)
            y = y.transpose(1, 2).contiguous().view(batch, length, c)
    y = self.resid_dropout(self.c_proj(y))
    return y, (kbuf[:, :, :past + length], vbuf[:, :, :past + length])


@contextmanager
def mode(model, name):
    saved = []
    try:
        for block in model.transformer.h:
            a = block.attn
            saved.append((a, a.forward, a.attention_backend))
            a.attention_backend = 'flash'
            if name != 'dynamic_flash':
                a._joint_mode, a._capacity = name, model.block_size
                a.forward = MethodType(_attention, a)
        yield
    finally:
        for a, forward, backend in saved:
            a.forward, a.attention_backend = forward, backend
            if hasattr(a, '_buffers_kv'):
                del a._buffers_kv


def merge(model):
    """Create the strong serving control, using FP32 to form the merged weights."""
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                w = child.linear.weight.float() + child.scale * (child.B.weight.float() @ child.A.weight.float())
                child.linear.weight.copy_(w.to(child.linear.weight.dtype))
                setattr(parent, name, child.linear)

"""Inference-only tiled online-softmax attention with IEEE FP32 dot products.

This is a small FlashAttention-algorithm prototype, not a port of PyTorch's
optimized FlashAttention backend. Triton is imported only when this module is used.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _tiled_fp32(Q, K, V, O,
                QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr, QS3: tl.constexpr,
                KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr, KS3: tl.constexpr,
                VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr, VS3: tl.constexpr,
                HEADS: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
                D: tl.constexpr, PAST: tl.constexpr, SCALE: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    block = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // HEADS
    head = bh % HEADS
    qi = block * BM + tl.arange(0, BM)
    di = tl.arange(0, BD)
    kj = tl.arange(0, BN)
    q = tl.load(Q + batch*QS0 + head*QS1 + qi[:, None]*QS2 + di[None, :]*QS3,
                mask=(qi[:, None] < NQ) & (di[None, :] < D), other=0)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, BD), tl.float32)
    # Only visit keys visible to at least one query in this tile.
    stop = tl.minimum(NK, PAST + (block+1)*BM)
    for start in range(0, stop, BN):
        ki = start + kj
        k = tl.load(K + batch*KS0 + head*KS1 + di[:, None]*KS3 + ki[None, :]*KS2,
                    mask=(di[:, None] < D) & (ki[None, :] < NK), other=0)
        scores = tl.dot(q, k, input_precision='ieee') * SCALE
        visible = (ki[None, :] < NK) & (ki[None, :] <= PAST + qi[:, None])
        scores = tl.where(visible, scores, -float('inf'))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        probabilities = tl.exp(scores - new_maximum[:, None])
        correction = tl.exp(maximum - new_maximum)
        v = tl.load(V + batch*VS0 + head*VS1 + ki[:, None]*VS2 + di[None, :]*VS3,
                    mask=(ki[:, None] < NK) & (di[None, :] < D), other=0)
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probabilities, v, input_precision='ieee')
        denominator = denominator * correction + tl.sum(probabilities, 1)
        maximum = new_maximum
    output = accumulator / denominator[:, None]
    tl.store(O + ((batch*HEADS + head)*NQ + qi[:, None])*D + di[None, :], output,
             mask=(qi[:, None] < NQ) & (di[None, :] < D))


def tiled_attention_fp32(q, k, v, past_length=0, *, return_kernel=False):
    """Dense equal-length causal attention; query positions start at past_length.

    Q/K/V are B,H,T,D FP32 CUDA tensors, with arbitrary strides. No dropout or
    backward is implemented. Both matrix products explicitly request IEEE FP32;
    softmax, its running state, and output also remain FP32.
    """
    if any(t.device.type != 'cuda' or t.dtype != torch.float32 for t in (q,k,v)):
        raise ValueError('tiled_fp32 requires FP32 CUDA Q/K/V')
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q,k,v)):
        raise RuntimeError('tiled_fp32 is inference-only; backward is not implemented')
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError('Q/K/V must have compatible B,H,T,D shapes')
    b,h,nq,d = q.shape
    if k.shape[:2] != (b,h) or k.shape[3] != d or k.size(2) != past_length+nq:
        raise ValueError('K/V length must equal past_length + query length')
    if not 1 <= d <= 128 or nq == 0:
        raise ValueError('head width must be 1..128 and query length positive')
    if k.device != q.device or v.device != q.device:
        raise ValueError('Q/K/V must share a device')
    out = torch.empty((b,h,nq,d), device=q.device, dtype=torch.float32)
    with torch.cuda.device(q.device):
        compiled = _tiled_fp32[(triton.cdiv(nq,16),b*h)](
            q,k,v,out,*q.stride(),*k.stride(),*v.stride(),h,nq,k.size(2),d,past_length,d**-.5,
            BM=16,BN=32,BD=max(16,triton.next_power_of_2(d)),num_warps=4)
    return (out, compiled) if return_kernel else out

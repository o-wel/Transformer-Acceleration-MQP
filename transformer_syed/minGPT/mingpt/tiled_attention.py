"""Tiled online-softmax attention with IEEE FP32 forward and backward dots.

This is a small FlashAttention-algorithm prototype, not a port of PyTorch's
optimized FlashAttention backend. Triton is imported only when this module is used.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _tiled_fp32(Q, K, V, O, LSE,
                QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr, QS3: tl.constexpr,
                KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr, KS3: tl.constexpr,
                VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr, VS3: tl.constexpr,
                HEADS: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
                D: tl.constexpr, PAST: tl.constexpr, SCALE: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr, STORE_LSE: tl.constexpr):
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
    if STORE_LSE:
        tl.store(LSE + bh*NQ + qi, maximum + tl.log(denominator), mask=qi < NQ)
    output = accumulator / denominator[:, None]
    tl.store(O + ((batch*HEADS + head)*NQ + qi[:, None])*D + di[None, :], output,
             mask=(qi[:, None] < NQ) & (di[None, :] < D))



@triton.jit
def _row_delta(O, DO, DELTA, N: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    di = tl.arange(0, BD)
    o = tl.load(O + row*D + di, mask=di < D, other=0)
    do = tl.load(DO + row*D + di, mask=di < D, other=0)
    tl.store(DELTA + row, tl.sum(o*do, 0))


@triton.jit
def _backward_q(Q, K, V, DO, LSE, DELTA, DQ,
                HEADS: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
                D: tl.constexpr, PAST: tl.constexpr, SCALE: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    block = tl.program_id(0)
    bh = tl.program_id(1)
    qi = block*BM + tl.arange(0, BM)
    ki = tl.arange(0, BN)
    di = tl.arange(0, BD)
    q = tl.load(Q + (bh*NQ + qi[:, None])*D + di[None, :],
                mask=(qi[:, None] < NQ) & (di[None, :] < D), other=0)
    do = tl.load(DO + (bh*NQ + qi[:, None])*D + di[None, :],
                 mask=(qi[:, None] < NQ) & (di[None, :] < D), other=0)
    lse = tl.load(LSE + bh*NQ + qi, mask=qi < NQ, other=0)
    delta = tl.load(DELTA + bh*NQ + qi, mask=qi < NQ, other=0)
    dq = tl.zeros((BM, BD), tl.float32)
    stop = tl.minimum(NK, PAST + (block+1)*BM)
    for start in range(0, stop, BN):
        keys = start + ki
        k = tl.load(K + (bh*NK + keys[:, None])*D + di[None, :],
                    mask=(keys[:, None] < NK) & (di[None, :] < D), other=0)
        v = tl.load(V + (bh*NK + keys[:, None])*D + di[None, :],
                    mask=(keys[:, None] < NK) & (di[None, :] < D), other=0)
        scores = tl.dot(q, tl.trans(k), input_precision='ieee')*SCALE
        visible = (qi[:, None] < NQ) & (keys[None, :] < NK) & (keys[None, :] <= PAST + qi[:, None])
        p = tl.exp(tl.where(visible, scores - lse[:, None], -float('inf')))
        dp = tl.dot(do, tl.trans(v), input_precision='ieee')
        ds = p*(dp - delta[:, None])
        dq += tl.dot(ds, k, input_precision='ieee')*SCALE
    tl.store(DQ + (bh*NQ + qi[:, None])*D + di[None, :], dq,
             mask=(qi[:, None] < NQ) & (di[None, :] < D))


@triton.jit
def _backward_kv(Q, K, V, DO, LSE, DELTA, DK, DV,
                 HEADS: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
                 D: tl.constexpr, PAST: tl.constexpr, SCALE: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    block = tl.program_id(0)
    bh = tl.program_id(1)
    keys = block*BN + tl.arange(0, BN)
    query_offsets = tl.arange(0, BM)
    di = tl.arange(0, BD)
    k = tl.load(K + (bh*NK + keys[:, None])*D + di[None, :],
                mask=(keys[:, None] < NK) & (di[None, :] < D), other=0)
    v = tl.load(V + (bh*NK + keys[:, None])*D + di[None, :],
                mask=(keys[:, None] < NK) & (di[None, :] < D), other=0)
    dk = tl.zeros((BN, BD), tl.float32)
    dv = tl.zeros((BN, BD), tl.float32)
    # Each program owns a K/V tile: no atomics or quadratic global buffers.
    for start in range(tl.maximum(0, block*BN - PAST), NQ, BM):
        qi = start + query_offsets
        q = tl.load(Q + (bh*NQ + qi[:, None])*D + di[None, :],
                    mask=(qi[:, None] < NQ) & (di[None, :] < D), other=0)
        do = tl.load(DO + (bh*NQ + qi[:, None])*D + di[None, :],
                     mask=(qi[:, None] < NQ) & (di[None, :] < D), other=0)
        lse = tl.load(LSE + bh*NQ + qi, mask=qi < NQ, other=0)
        delta = tl.load(DELTA + bh*NQ + qi, mask=qi < NQ, other=0)
        scores = tl.dot(k, tl.trans(q), input_precision='ieee')*SCALE
        visible = (keys[:, None] < NK) & (qi[None, :] < NQ) & (keys[:, None] <= PAST + qi[None, :])
        p = tl.exp(tl.where(visible, scores - lse[None, :], -float('inf')))
        dp = tl.dot(v, tl.trans(do), input_precision='ieee')
        ds = p*(dp - delta[None, :])
        dk += tl.dot(ds, q, input_precision='ieee')*SCALE
        dv += tl.dot(p, do, input_precision='ieee')
    tl.store(DK + (bh*NK + keys[:, None])*D + di[None, :], dk,
             mask=(keys[:, None] < NK) & (di[None, :] < D))
    tl.store(DV + (bh*NK + keys[:, None])*D + di[None, :], dv,
             mask=(keys[:, None] < NK) & (di[None, :] < D))


def _forward(q, k, v, past_length, save_lse):
    b,h,nq,d = q.shape
    out = torch.empty((b,h,nq,d), device=q.device, dtype=torch.float32)
    lse = torch.empty((b,h,nq), device=q.device, dtype=torch.float32) if save_lse else out
    with torch.cuda.device(q.device):
        compiled = _tiled_fp32[(triton.cdiv(nq,16),b*h)](
            q,k,v,out,lse,*q.stride(),*k.stride(),*v.stride(),h,nq,k.size(2),d,past_length,d**-.5,
            BM=16,BN=32,BD=max(16,triton.next_power_of_2(d)),STORE_LSE=save_lse,num_warps=4)
    return out,lse,compiled


def _run_backward(q, k, v, out, lse, grad, past_length, return_kernels=False):
    # Copies are linear in sequence length. They handle packed QKV and arbitrary
    # upstream gradient strides; their cost is included in training measurements.
    q,k,v,grad = [x.contiguous() for x in (q,k,v,grad)]
    b,h,nq,d = q.shape
    nk = k.size(2)
    dq,dk,dv = [torch.empty_like(x) for x in (q,k,v)]
    delta = torch.empty_like(lse)
    bd = max(16,triton.next_power_of_2(d))
    with torch.cuda.device(q.device):
        kd = _row_delta[(b*h*nq,)](out,grad,delta,nq,d,bd,num_warps=4)
        kq = _backward_q[(triton.cdiv(nq,16),b*h)](
            q,k,v,grad,lse,delta,dq,h,nq,nk,d,past_length,d**-.5,16,32,bd,num_warps=4)
        kkv = _backward_kv[(triton.cdiv(nk,16),b*h)](
            q,k,v,grad,lse,delta,dk,dv,h,nq,nk,d,past_length,d**-.5,32,16,bd,num_warps=4)
    grads = (dq,dk,dv)
    return (grads,{'delta':kd,'dq':kq,'dkdv':kkv}) if return_kernels else grads


class _TiledAttentionFP32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, past_length):
        out,lse,_ = _forward(q,k,v,past_length,True)
        ctx.save_for_backward(q,k,v,out,lse)
        ctx.past_length = past_length
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        q,k,v,out,lse = ctx.saved_tensors
        dq,dk,dv = _run_backward(q,k,v,out,lse,grad,ctx.past_length)
        return dq,dk,dv,None


def tiled_attention_fp32(q, k, v, past_length=0, *, return_kernel=False):
    """Causal FP32 CUDA attention, with first-order tiled backward and no dropout.

    Q/K/V are B,H,T,D FP32 CUDA tensors, including noncontiguous packed QKV.
    Queries begin at past_length. Gradients are supported for all three inputs;
    higher-order derivatives are not. Both forward/backward dot products request
    IEEE FP32. Softmax uses FP32 GPU exponential/log approximations.
    """
    if any(t.device.type != 'cuda' or t.dtype != torch.float32 for t in (q,k,v)):
        raise ValueError('tiled_fp32 requires FP32 CUDA Q/K/V')
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError('Q/K/V must have compatible B,H,T,D shapes')
    if not isinstance(past_length,int) or isinstance(past_length,bool) or past_length < 0:
        raise ValueError('past_length must be a nonnegative integer')
    b,h,nq,d = q.shape
    if k.shape[:2] != (b,h) or k.shape[3] != d or k.size(2) != past_length+nq:
        raise ValueError('K/V length must equal past_length + query length')
    if not 1 <= d <= 128 or nq == 0 or b == 0 or h == 0:
        raise ValueError('head width must be 1..128 and batch/heads/query length positive')
    if k.device != q.device or v.device != q.device:
        raise ValueError('Q/K/V must share a device')
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q,k,v)):
        if return_kernel:
            raise ValueError('return_kernel is only available without autograd')
        return _TiledAttentionFP32.apply(q,k,v,past_length)
    out,_,compiled = _forward(q,k,v,past_length,False)
    return (out,compiled) if return_kernel else out

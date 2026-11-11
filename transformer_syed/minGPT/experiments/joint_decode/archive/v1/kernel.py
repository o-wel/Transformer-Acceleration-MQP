"""Experimental one-token LoRA expansion + cache append + online attention.

This is an original Triton prototype of a candidate fusion, not an official
FlashAttention implementation. Historical cache entries are fully adapted.
Only the current token's LoRA expansion is fused; A and base GEMMs stay outside.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(QKV, Z, UP, KC, VC, O, C: tl.constexpr, H: tl.constexpr,
            D: tl.constexpr, R: tl.constexpr, RR: tl.constexpr,
            CAP: tl.constexpr, PAST, SCALE: tl.constexpr,
            FUSED: tl.constexpr, BN: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    ds = tl.arange(0, D)
    cols = head * D + ds
    q = tl.load(QKV + batch * 3 * C + cols)
    k = tl.load(QKV + batch * 3 * C + C + cols)
    v = tl.load(QKV + batch * 3 * C + 2 * C + cols)
    if FUSED:
        rs = tl.arange(0, RR)
        z = tl.load(Z + batch * R + rs, rs < R, 0).to(tl.float32)
        bq = tl.load(UP + cols[:, None] * R + rs[None, :], rs[None, :] < R, 0).to(tl.float32)
        bk = tl.load(UP + (C + cols[:, None]) * R + rs[None, :], rs[None, :] < R, 0).to(tl.float32)
        bv = tl.load(UP + (2 * C + cols[:, None]) * R + rs[None, :], rs[None, :] < R, 0).to(tl.float32)
        # Preserve the eager FP16 multiply/add rounding boundaries. Reduction
        # order may differ; correctness is tolerance-based, not bitwise claimed.
        dq = (tl.sum(bq * z[None, :], 1).to(q.dtype).to(tl.float32) * SCALE).to(q.dtype)
        dk = (tl.sum(bk * z[None, :], 1).to(k.dtype).to(tl.float32) * SCALE).to(k.dtype)
        dv = (tl.sum(bv * z[None, :], 1).to(v.dtype).to(tl.float32) * SCALE).to(v.dtype)
        q = (q.to(tl.float32) + dq.to(tl.float32)).to(q.dtype)
        k = (k.to(tl.float32) + dk.to(tl.float32)).to(k.dtype)
        v = (v.to(tl.float32) + dv.to(tl.float32)).to(v.dtype)
    base = (batch * H + head) * CAP * D
    tl.store(KC + base + PAST * D + ds, k)
    tl.store(VC + base + PAST * D + ds, v)
    # Start online softmax with the current token, kept in registers. No CTA
    # reads another CTA's new cache write, so append needs no global barrier.
    qf = q.to(tl.float32)
    m = tl.sum(qf * k.to(tl.float32), 0) * (D ** -0.5)
    denom = 1.0
    acc = v.to(tl.float32)
    ns = tl.arange(0, BN)
    for start in range(tl.cdiv(PAST, BN)):
        ts = start * BN + ns
        keys = tl.load(KC + base + ts[:, None] * D + ds[None, :], ts[:, None] < PAST, 0).to(tl.float32)
        vals = tl.load(VC + base + ts[:, None] * D + ds[None, :], ts[:, None] < PAST, 0).to(tl.float32)
        scores = tl.sum(keys * qf[None, :], 1) * (D ** -0.5)
        scores = tl.where(ts < PAST, scores, -float('inf'))
        newm = tl.maximum(m, tl.max(scores, 0))
        correction = tl.exp(m - newm)
        probs = tl.exp(scores - newm)
        acc = acc * correction + tl.sum(probs[:, None] * vals, 0)
        denom = denom * correction + tl.sum(probs, 0)
        m = newm
    tl.store(O + batch * C + cols, acc / denom)


def decode(qkv, cache, past, heads, *, z=None, up=None, scale=1.0):
    assert qkv.ndim == 3 and qkv.size(1) == 1 and qkv.dtype == torch.float16
    assert qkv.is_contiguous() and qkv.device.type == 'cuda'
    batch, _, three_c = qkv.shape
    c = three_c // 3
    k, v = cache
    assert c % heads == 0 and triton.next_power_of_2(c // heads) == c // heads
    assert k.is_contiguous() and v.is_contiguous() and k.shape == v.shape
    assert k.dtype == v.dtype == qkv.dtype and 0 <= past < k.size(2)
    fused = z is not None
    rank = z.size(-1) if fused else 1
    if fused:
        assert up.shape == (three_c, rank) and up.is_contiguous() and z.is_contiguous()
    out = torch.empty((batch, 1, c), device=qkv.device, dtype=qkv.dtype)
    _decode[(batch, heads)](qkv, z if fused else qkv, up if fused else qkv,
        k, v, out, c, heads, c // heads, rank, triton.next_power_of_2(rank),
        k.size(2), past, scale, fused, 128, num_warps=4, enable_fp_fusion=False)
    return out

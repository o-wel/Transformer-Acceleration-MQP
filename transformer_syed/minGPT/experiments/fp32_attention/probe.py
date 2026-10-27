import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import unittest

torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
print('GPU',torch.cuda.get_device_name(), 'torch',torch.__version__,flush=True)
suite=unittest.defaultTestLoader.discover('tests',pattern='test_flash_attention.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise RuntimeError('GPU correctness tests failed')
from mingpt.tiled_attention import tiled_attention_fp32
from mingpt.model import GPT
q=torch.randn(1,12,512,64,device='cuda')
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        F.scaled_dot_product_attention(q,q,q,is_causal=True)
prof.export_chrome_trace('experiments/fp32_attention/probe-trace.json')
print([(e.key,e.device_time_total) for e in prof.key_averages() if 'attention' in e.key],flush=True)
with torch.no_grad():
    for nq,past in [(37,0),(1,128),(7,128),(129,0)]:
        q=torch.randn(2,3,nq,64,device='cuda')
        k=torch.randn(2,3,nq+past,64,device='cuda')
        v=torch.randn_like(k)
        got=tiled_attention_fp32(q,k,v,past)
        mask=torch.arange(nq+past,device='cuda')[None,:] <= past+torch.arange(nq,device='cuda')[:,None]
        with sdpa_kernel(SDPBackend.MATH):
            ref=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        torch.testing.assert_close(got,ref,atol=2e-5,rtol=2e-5)
        print('TILED',nq,past,'max_error',float((got-ref).abs().max()),flush=True)
print('PROBE COMPLETE',flush=True)

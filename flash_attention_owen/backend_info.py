import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

q = torch.randn(
    1, 8, 512, 64,
    device="cuda",
    dtype=torch.bfloat16,
)
k = torch.randn_like(q)
v = torch.randn_like(q)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with torch.no_grad():
        F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=True,
        )

print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=30,
))
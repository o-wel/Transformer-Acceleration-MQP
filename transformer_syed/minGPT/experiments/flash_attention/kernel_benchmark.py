"""Attention-only BF16 latency: CUDA graph replay removes per-op Python dispatch."""
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

OUT = Path(__file__).resolve().parent/'results'


@torch.inference_mode()
def main():
    path = OUT/'attention-kernels.json'
    if path.exists():
        raise RuntimeError('Refusing to overwrite kernel measurements')
    torch.manual_seed(777)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  node=platform.node(), job=os.environ.get('SLURM_JOB_ID'), dtype='bfloat16',
                  graph_operations=20, repeats=15, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=[])
    for batch, length in [(1,32),(1,128),(1,512),(4,128),(1,896)]:
        for phase in ['prefill','decode']:
            qlen, klen = (length,length) if phase == 'prefill' else (1,length+63)
            q = torch.randn(batch,12,qlen,64,device='cuda',dtype=torch.bfloat16)
            k = torch.randn(batch,12,klen,64,device='cuda',dtype=torch.bfloat16)
            v = torch.randn_like(k)
            mask = torch.ones(qlen,klen,device='cuda',dtype=torch.bool).tril() if phase=='prefill' else None
            rows = {}
            for backend in ['eager','math','flash']:
                context = nullcontext() if backend == 'eager' else sdpa_kernel(SDPBackend.MATH if backend=='math' else SDPBackend.FLASH_ATTENTION)
                def attention():
                    if backend != 'eager':
                        return F.scaled_dot_product_attention(q,k,v,is_causal=phase=='prefill')
                    scores = (q @ k.transpose(-2,-1)) / math.sqrt(64)
                    if mask is not None:
                        scores = scores.masked_fill(~mask,float('-inf'))
                    return F.softmax(scores,dim=-1) @ v
                with context:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(10):
                            output = attention()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    del output
                    gc.collect()
                    torch.cuda.empty_cache()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    output = attention()
                    torch.cuda.synchronize()
                    memory = torch.cuda.max_memory_allocated()-baseline
                    del output
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(20):
                            output = attention()
                    for _ in range(5):
                        graph.replay()
                    torch.cuda.synchronize()
                    samples = []
                    for _ in range(15):
                        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end)*1000/20)
                    rows[backend] = dict(samples_us=samples, median_us=statistics.median(samples),
                                         mean_us=statistics.mean(samples),stdev_us=statistics.stdev(samples),
                                         incremental_peak_bytes=memory)
                    del graph, output
                gc.collect()
                torch.cuda.empty_cache()
            result['rows'].append(dict(batch=batch,prompt_tokens=length,query_tokens=qlen,key_tokens=klen,
                                       phase=phase,backends=rows,speedup=rows['eager']['median_us']/rows['flash']['median_us']))
            print('KERNEL',batch,length,phase,{b:round(r['median_us'],2) for b,r in rows.items()},flush=True)
            del q,k,v,mask
    path.write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

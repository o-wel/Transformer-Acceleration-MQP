"""Frozen design constants; no results-dependent grid expansion."""
import itertools

SEARCH_SEEDS = [11, 22, 33]
CONFIRM_SEEDS = [101, 202, 303, 404, 505]
BASE = dict(rank=4, alpha=8, lr=0.0001)
GRID = [dict(rank=r, alpha=r*s, lr=lr)
        for r,s,lr in itertools.product([2,4,8,16,32], [1,2,4], [0.0001,0.0003,0.001])]
CHUNKS = [32,64,128,256,512,1024]  # 1024 means a single full prefill for every workload.
TILES = list(itertools.product([16,32,64], [16,32,64,128], [4,8]))
BASE_CHUNK = 128
BASE_TILE = [16,32,4]
SEARCH_WORKLOADS = [dict(batch=b,prompt=p,new_tokens=64,split='search')
                    for b,p in itertools.product([1,4],[32,256,896])]
HELDOUT_WORKLOADS = [dict(batch=b,prompt=p,new_tokens=n,split='heldout')
                     for b,p,n in itertools.product([1,4],[128,512],[32,128])]
WORKLOADS = SEARCH_WORKLOADS+HELDOUT_WORKLOADS
DESIGN = dict(version=2, model='gpt2', precision='FP32 stored weights; FP16 autocast; TF32 disabled',
              search_seeds=SEARCH_SEEDS, confirmation_seeds=CONFIRM_SEEDS,
              lora_grid=GRID, baseline_lora=BASE, cache_chunks=CHUNKS, fa_tiles=TILES,
              baseline_chunk=BASE_CHUNK, baseline_tile=BASE_TILE,
              train=dict(steps=2400,batch=8,length=128,validate_every=300,warm_steps=50,
                         optimizer='AdamW',weight_decay=0,clip_norm=1.,lr_schedule='constant',
                         attention='eager',adapter_merged=False,dropout=0.,scaler_initial=128),
              search_workloads=SEARCH_WORKLOADS, heldout_workloads=HELDOUT_WORKLOADS,
              timing=dict(warmups=5,search_repeats=10,confirmation_repeats=20,
                          search_sessions=3,confirmation_sessions_per_seed=3,
                          metric='synchronized wall latency; CUDA event latency also recorded',
                          order='randomized paired blocks; same prompt per block; fresh prompt per repeat',
                          weighting='equal weight per fixed workload in log space'),
              selection=dict(lora='Minimum mean warm training-step latency among candidates whose mean validation-selected NLL across all three search seeds is within 0.01 of the best; retain quality-best separately',
                             runtime='Minimum equally weighted geometric mean wall latency across all six search workloads and three discovery allocations, subject to quality gates in every allocation',
                             checkpoint='Lowest validation answer-token NLL among 8 checkpoints; earliest step on exact tie',
                             ties='LoRA: lower rank, alpha, LR; runtime: original grid order',
                             no_test_selection=True),
              quality=dict(kernel_atol=.012,kernel_rtol=.04,cache_logit_atol=.1,cache_logit_rtol=.02,
                           max_abs_nll_delta=.01,lora_nll_margin=.01,decode_tokens=64,
                           primary='normalized exact reference match',secondary='primary and numeric fields; paired counterfactual fields',
                           independence_unit='match, including its three questions and score-swapped variants'),
              stats=dict(bootstrap_draws=5000,seed=20261105,
                         timing='Hierarchical paired bootstrap over 5 training seeds, 3 allocations within each seed, and 20 paired repeats within each workload; workloads fixed, equally weighted',
                         quality='Crossed paired bootstrap over 5 training seeds and 300 independent test matches; original and counterfactual questions clustered together',
                         primary='Combined all-tuned/all-untuned latency ratio on the 8 held-out workloads; 95% CI',
                         secondary='Per-workload and all-workload ratios, isolated methods, parameter factorial interactions, quality and training cost; descriptive intervals'),
              implementation='Locally snapshotted FA1-derived Triton prototype; not original FA1 CUDA or PyTorch FA2',
              scope='Prototype tuning only; no production FlashAttention reference',
              data=dict(train='Existing 480 matches / 1440 questions',validation='Fresh 200 matches / 600 questions',
                        test='Fresh 300 matches / 900 questions plus 900 paired counterfactuals',
                        generator_seed=20261105,heldout_teams=['Badgers','Panthers','Herons','Otters','Foxes','Ravens','Stallions','Cranes']),
              policy='No pilot timings or test scores pooled. No claim of global optimality, cross-model, cross-task, or cross-hardware generalization. Untuned control predeclared, never selected as worst. No joint tuning after freezing individual choices.')


def config_key(c):
    return f"r{c['rank']}-a{c['alpha']}-lr{c['lr']:g}"


def workload_key(w):
    return f"b{w['batch']}-p{w['prompt']}-n{w['new_tokens']}"

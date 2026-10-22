"""Check recorded KV experiment completeness, controls, and internal consistency."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'results'
r=json.loads((OUT/'report.json').read_text())
assert r['complete'] and r['dtype']=='float32' and not r['tf32']
for path,expected in r['source_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
for name,info in r['models'].items():
    assert info['weights_unchanged'] and info['weight_sha256_before']==info['weight_sha256_after']
    quality=json.loads((OUT/f'{name}-quality.json').read_text())
    assert quality['questions']==quality['identical_all_40_tokens']==quality['identical_until_eos']==quality['identical_texts']==120
    assert quality['cached']==quality['uncached']
    assert len(quality['outputs'])==120
    for output in quality['outputs']:
        assert output['cached']==output['uncached']
        assert output['cached_scores']==output['uncached_scores']
    if name=='full':
        continue
    assert len(info['performance'])==4
    for row in info['performance']:
        assert row['new_tokens']==64 and row['repeats']==5
        assert row['prompt_tokens']+63<=1024
        for mode,samples in row['samples'].items():
            assert len(samples)==5
            for sample in samples:
                assert all(math.isfinite(v) and v>=0 for v in sample.values())
                assert sample['peak_reserved_bytes']>=sample['peak_allocated_bytes']>=sample['baseline_allocated_bytes']
                expected=0 if mode=='uncached' else 2*12*row['batch_size']*(row['prompt_tokens']+63)*768*4
                assert sample['cache_bytes']==expected
                assert math.isclose(sample['aggregate_tokens_per_second'],row['batch_size']*64/(sample['wall_ms']/1000))
                assert math.isclose(sample['decode_ms_per_token_step'],sample['decode_ms']/63)
            for key,stats in row['summary'][mode].items():
                assert math.isclose(stats['mean'],statistics.mean(s[key] for s in samples))
        assert math.isclose(row['speedup'],row['summary']['uncached']['wall_ms']['mean']/row['summary']['cached']['wall_ms']['mean'])
print('PASS: FP32 controls, source hashes, 360 identical outputs, unchanged weights, timing arithmetic, cache size, and all 80 measured requests.')

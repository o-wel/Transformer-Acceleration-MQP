"""Pure protocol definitions for a three-method optimization ablation."""
import itertools
import json
import hashlib

STATES = ('off', 'best', 'poor')
SEEDS = (42, 43, 44)
WORKLOADS = [dict(batch=b, prompt_tokens=p, new_tokens=n)
             for b, p, n in itertools.product((1, 4, 8), (32, 128, 512, 896), (32, 64))]
ATTENTION = [dict(backend='eager', bm=16, bn=32, warps=4)] + [
    dict(backend='fa1', bm=m, bn=n, warps=w)
    for m, n, w in itertools.product((16, 32), (32, 64), (4, 8))]
CACHES = ('off', 'dynamic', 'static_request', 'static_context')


def uid(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def states():
    return [dict(zip(('lora', 'attention', 'cache'), s)) for s in itertools.product(STATES, repeat=3)]


def choose(rows):
    """Pick extrema by calibration latency among passing cells; keep failures diagnostic."""
    if not rows:
        return dict(best=None, poor=None, eligible=False, reason='No timed candidates')
    passed = [r for r in rows if r['status'] == 'ok']
    candidates = passed or [r for r in rows if r.get('wall_ms') is not None]
    if not candidates:
        return dict(best=None, poor=None, eligible=False, reason='All candidates failed execution')
    ordered = sorted(candidates, key=lambda r: (r['wall_ms'], uid(r['config'])))
    return dict(best=ordered[0]['config'], poor=ordered[-1]['config'], eligible=bool(passed),
                reason='Passing calibration extrema' if passed else 'Diagnostic only: no candidate passed parity',
                passing_candidates=len(passed), distinct=len({uid(r['config']) for r in candidates}) > 1)


def selections(rows, lora, workload):
    group = [r for r in rows if r['lora']==lora and r['workload']==workload]
    return dict(attention=choose([r for r in group if r['config']['backend']=='fa1' and r['config']['cache_mode']=='off']),
                cache=choose([r for r in group if r['config']['backend']=='eager' and r['config']['cache_mode']!='off']),
                joint=choose([r for r in group if r['config']['backend']=='fa1' and r['config']['cache_mode']!='off']))


def materialize(state, selection):
    a = dict(backend='eager', bm=16, bn=32, warps=4)
    c = 'off'
    if state['attention']!='off':
        pick = selection['attention'][state['attention']]
        if pick is None: return None
        a = {k:pick[k] for k in a}
    if state['cache']!='off':
        pick = selection['cache'][state['cache']]
        if pick is None: return None
        c = pick['cache_mode']
    return dict(**a, cache_mode=c)

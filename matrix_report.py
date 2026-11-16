"""Build explicit coverage, raw CSVs, and a paginated PDF without optional packages."""
import collections
import csv
import html
import json
from pathlib import Path
import re
import statistics
import textwrap
from protocol import states, WORKLOADS, SEEDS, uid


def pdf(path, lines):
    pages=[];page=[]
    for raw in lines:
        for line in textwrap.wrap(raw,width=105,replace_whitespace=False) or ['']:
            if len(page)==58:pages.append(page);page=[]
            page.append(line)
    if page:pages.append(page)
    objs=[b'<< /Type /Catalog /Pages 2 0 R >>',None,b'<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>'];kids=[]
    for i,lines in enumerate(pages,1):
        pid=len(objs)+1;sid=pid+1;kids.append(pid)
        objs.append(f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 3 0 R >> >> /Contents {sid} 0 R >>'.encode())
        ops=['0.06 0.16 0.25 rg BT /F1 8.6 Tf 12 TL 40 800 Td']
        for j,line in enumerate(lines):
            line=line.encode('ascii','replace').decode().replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
            if j:ops.append('T*')
            ops.append(f'({line}) Tj')
        ops.append(f'ET BT /F1 8 Tf 40 55 Td (Optimization matrix | page {i}/{len(pages)}) Tj ET')
        stream='\n'.join(ops).encode();objs.append(f'<< /Length {len(stream)} >>\nstream\n'.encode()+stream+b'\nendstream')
    objs[1]=f'<< /Type /Pages /Kids [{" ".join(f"{p} 0 R" for p in kids)}] /Count {len(kids)} >>'.encode()
    buf=bytearray(b'%PDF-1.4\n');offsets=[]
    for i,obj in enumerate(objs,1):offsets.append(len(buf));buf.extend(f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n')
    pos=len(buf);buf.extend(f'xref\n0 {len(objs)+1}\n0000000000 65535 f \n'.encode())
    for off in offsets:buf.extend(f'{off:010d} 00000 n \n'.encode())
    buf.extend(f'trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{pos}\n%%EOF\n'.encode());path.write_bytes(buf)


def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def build(out):
    read=lambda name:json.loads((out/name).read_text())
    collect=lambda pattern:sum([json.loads(p.read_text()) for p in sorted(out.glob(pattern))],[])
    plan=read('plan.json');cal=collect('calibration-*.json');conf=collect('confirmation-*.json');quality=collect('quality-*.json')
    training=read('training.json') if (out/'training.json').exists() else {}
    expected=dict(calibration=2592,confirmation=2232,quality=93,training=9)
    actual=dict(calibration=len(cal),confirmation=len(conf),quality=len(quality),training=len(training.get('results',[])))
    repetitions_ok=all(r['status'] in ('oom','unavailable') or len(r.get('samples',[]))==7 for r in conf)
    unique_conf=len({r['id'] for r in conf})==len(conf)
    unique_cal=len({r['id'] for r in cal})==len(cal)
    unique_quality=len({(r['seed'],uid(r['state'])) for r in quality})==len(quality)
    complete=(actual==expected and repetitions_ok and unique_conf and unique_cal and unique_quality
              and (out/'serving_selection.json').exists() and (out/'matrix_smoke.json').exists()
              and all(r.get('complete') and r.get('frozen_base_unchanged') for r in training.get('results',[])))
    state='COMPLETE' if complete else 'PARTIAL'
    status=dict(status=state,expected=expected,actual=actual,confirmation_status=dict(collections.Counter(r['status'] for r in conf)),
                repetitions_ok=repetitions_ok)
    (out/'REPORT_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    flat=[]
    for r in conf:
        row=dict(seed=r['seed'],**r['state'],**r['workload'],**(r.get('config') or {}),status=r['status'],
                 logits_pass=r.get('logits_pass'),tokens_equal=r.get('tokens_equal'),wall_ms=r.get('wall_ms'))
        if r['samples']:
            for key in r['samples'][0]:
                row[key+'_median']=statistics.median(x[key] for x in r['samples'])
                row[key+'_sd']=statistics.stdev(x[key] for x in r['samples'])
        flat.append(row)
    csv_write(out/'CONFIRMATION.csv',flat)
    csv_write(out/'CALIBRATION.csv',[dict(lora=r['lora'],**r['workload'],**r['config'],status=r['status'],wall_ms=r['wall_ms']) for r in cal])
    csv_write(out/'QUALITY.csv',[dict(seed=r['seed'],**r['state'],status=r['status'],validation_nll=r.get('validation_nll'),**(r.get('metrics') or {})) for r in quality])
    csv_write(out/'TRAINING.csv',[dict(label=r['label'],seed=r['seed'],**r['config'],validation_nll=r['validation_nll'],
        peak_training_bytes=r['peak_training_bytes'],trainable_parameters=r['trainable_parameters'],median_step_ms=r['median_step_ms'],
        training_update_seconds=sum(r['step_ms'])/1000) for r in training.get('results',[])])
    lines=[f'# Optimization matrix: {plan["task"]} / {plan["precision"]}',f'Status: {state}', '',
        'Design: 27 off/best/poor combinations plus four jointly selected attention/cache cases.',
        'Best and poor are observed finite-grid choices. Training configurations were fixed from prior validation.',
        'Poor LoRA was selected by worst seed-42 full-budget validation loss, then retrained on three seeds.',
        'Off LoRA means the selected full-fine-tuning control; this is not an unadapted pretrained model.',
        'Attention/cache best and poor are calibrated independently per checkpoint and workload; joint cases use paired settings.',
        'Calibration uses a distinct prompt stream and three timings; confirmation uses seven timings and three training seeds.',
        'Strict parity is checked against each checkpoint on eager/no-cache, not between different trained checkpoints.',
        'Failures remain diagnostic. Complete artifact coverage does not mean every optimization passed correctness.',
        'Task quality uses natural held-out examples with canonical serving choices (batch 1, prompt 128, output 32).',
        'SMS is a one-step label task; its cache benefit is assessed only by synthetic serving workloads.',
        'Timing SD within an allocation and variation across model seeds are not independent hardware replications.',
        'Original data splits are reused; full/LoRA inherited tuning search budgets are unequal.',
        '', 'Coverage: '+json.dumps(status,sort_keys=True), '', '## Frozen training configurations']
    for label,c in plan['configs'].items():lines.extend([label+': '+json.dumps(c,sort_keys=True)])
    lines.extend(['','## Task quality by state (mean across available seeds)','',
                  'LoRA / Attention / Cache | seeds | metrics'])
    qs=collections.defaultdict(list)
    for r in quality:qs[tuple(r['state'].values())].append(r)
    for key,rr in sorted(qs.items()):
        metrics={k:statistics.mean(r['metrics'][k] for r in rr if r.get('metrics') and k in r['metrics'])
                 for k in dict.fromkeys(k for r in rr for k in (r.get('metrics') or {})) if k not in ('tp','tn','fp','fn','n')}
        lines.append(' / '.join(key)+f' | {len(rr)}/3 | '+', '.join(f'{k}={v:.5g}' for k,v in metrics.items()))
    lines.extend(['','## Confirmation by workload','',
                  'Latency and peak memory are medians across seed medians. Speed ratio uses the all-off control.',
                  'Each row reports passing seeds; partial passing coverage cannot support a blanket speed claim.'])
    summary=[]
    for workload in WORKLOADS:
        group=[r for r in conf if r['workload']==workload]
        if not group:continue
        lines.extend(['',f'Batch {workload["batch"]}, prompt {workload["prompt_tokens"]}, output {workload["new_tokens"]}',
                      'LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio'])
        baseline={r['seed']:r for r in group if all(v=='off' for v in r['state'].values())}
        groups=collections.defaultdict(list)
        for r in group:groups[tuple(r['state'].values())].append(r)
        for key,rr in sorted(groups.items()):
            timed=[r for r in rr if r.get('wall_ms') is not None]
            passing=[r for r in rr if r['status']=='ok']
            latency=statistics.median(r['wall_ms'] for r in timed) if timed else None
            memory=statistics.median(statistics.median(s['peak_allocated_bytes'] for s in r['samples']) for r in timed) / 2**20 if timed else None
            ratios=[baseline[r['seed']]['wall_ms']/r['wall_ms'] for r in passing if r['seed'] in baseline and baseline[r['seed']]['status']=='ok']
            ratio=statistics.median(ratios) if len(ratios)==3 and {r['seed'] for r in passing}==set(SEEDS) else None
            row=dict(**workload,lora=key[0],attention=key[1],cache=key[2],passing_seeds=len(passing),completed_seeds=len(rr),
                     median_ms=latency,peak_MiB=memory,all_seeds_passing_speed_ratio=ratio)
            summary.append(row)
            fmt=lambda v:'-' if v is None else f'{v:.3f}'
            lines.append(' / '.join(key)+f' | {len(passing)}/{len(rr)} | {fmt(latency)} | {fmt(memory)} | {fmt(ratio)}')
    csv_write(out/'COMPARISONS.csv',summary)
    lines+=['','## Source and provenance',f'Study: {out.parent.name}',
            'Source and dataset hashes: ../SNAPSHOT_SHA256.json; frozen plans include prior selection/trial hashes.',
            'Raw timings, numerical errors and dtype audits: calibration-*.json and confirmation-*.json.',
            'Natural-task outputs: quality-*.json. Training and checkpoint hashes: training.json.',
            'No test score is used to select configurations. Diagnostic results are retained.']
    (out/'PERFORMANCE_REVIEW.md').write_text('\n'.join(lines)+'\n')
    (out/'PERFORMANCE_REVIEW.html').write_text('<!doctype html><meta charset="utf-8"><title>Optimization matrix</title><style>body{max-width:1100px;margin:40px auto;font:14px/1.6 monospace}pre{white-space:pre-wrap}</style><pre>'+html.escape('\n'.join(lines))+'</pre>')
    pdf(out/'PERFORMANCE_REVIEW.pdf',lines)
    print(json.dumps(status),flush=True)

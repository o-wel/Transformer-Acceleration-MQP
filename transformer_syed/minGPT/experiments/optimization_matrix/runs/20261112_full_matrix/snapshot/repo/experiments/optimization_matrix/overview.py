"""Aggregate honest coverage and links for the four task/precision studies."""
import json
from pathlib import Path
from report import pdf

def build(root):
    lines=['# LoRA, attention and KV-cache optimization matrix','',
           'Each task/precision contains 27 off/best/poor combinations plus four joint serving selections.',
           'Both independently tuned factors and jointly tuned attention/cache settings are confirmed.',
           'Off LoRA is full fine-tuning; attention and cache are inference factors in this study.',
           'Best/poor training choices are frozen from prior validation. Numerical failures stay diagnostic.',
           'New measurements are not available until jobs complete. Submission is not completion.','']
    statuses={}
    for name in ['basketball_fp16','basketball_fp32','spam_fp16','spam_fp32']:
        run=root/name;p=run/'REPORT_STATUS.json'
        status=json.loads(p.read_text()) if p.exists() else dict(status='PARTIAL',actual={})
        statuses[name]=status
        lines += [f'## {name}: {status["status"]}',json.dumps(status.get('actual',{}),sort_keys=True),
                  f'Report: {name}/PERFORMANCE_REVIEW.pdf',f'Raw comparison table: {name}/COMPARISONS.csv','']
    complete=all(v['status']=='COMPLETE' for v in statuses.values())
    lines.insert(2,'Overall status: '+('COMPLETE' if complete else 'PARTIAL'))
    lines += ['The four studies retune separately. Precision contrasts between selected winners combine precision',
              'and configuration changes; do not interpret them as isolated FP16 effects.',
              'Input provenance and frozen source/data hashes: SNAPSHOT_SHA256.json. Scheduler IDs: jobs.json.']
    (root/'PERFORMANCE_REVIEW.md').write_text('\n'.join(lines)+'\n');pdf(root/'PERFORMANCE_REVIEW.pdf',lines)
    (root/'REPORT_STATUS.json').write_text(json.dumps(dict(status='COMPLETE' if complete else 'PARTIAL',studies=statuses),indent=2)+'\n')

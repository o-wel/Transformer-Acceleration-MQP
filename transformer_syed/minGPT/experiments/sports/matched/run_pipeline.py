"""Equal tuning budget, selection by validation only, then three-seed evaluation."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path('experiments/sports/matched')
RATES=[0.001,0.0001]


def run(path, *args):
    subprocess.run([sys.executable,'-u','sports_experiment.py','--out',str(path),*args],check=True)


def select_candidate(candidates):
    return min(candidates,key=lambda x:(x['validation_nll'],x['lr']))


def main():
    manifest_path=ROOT/'selection.json'
    if manifest_path.exists():
        raise FileExistsError('Existing matched results must be preserved; use new output paths.')
    manifest=dict(protocol='Two learning rates per method, seed 42, 1200 steps each. Select by minimum validation NLL; then evaluate and run seeds 43/44. No test-guided selection.',rates=RATES,methods={})
    for method in ['lora','full']:
        candidates=[]
        for rate in RATES:
            path=Path(f'out/matched-{method}-lr{rate:g}-seed42')
            run(path,'--seed','42','--lr',str(rate),'--skip-test',*(['--full'] if method=='full' else []))
            report=json.loads((path/'report.json').read_text())
            assert not any(k in report for k in ['lora','full','base'])
            candidates.append(dict(path=str(path),lr=rate,validation_nll=report['selected_validation_nll'],step=report['selected_step']))
        chosen=select_candidate(candidates)
        manifest['methods'][method]=dict(candidates=candidates,selected_lr=chosen['lr'],runs={'42':chosen['path']})
    # Freeze both choices before ANY new test evaluation.
    manifest_path.write_text(json.dumps(manifest,indent=2))
    for method in ['lora','full']:
        info=manifest['methods'][method]
        run(info['runs']['42'],'--evaluate-only')
        for seed in [43,44]:
            path=Path(f'out/matched-{method}-seed{seed}')
            run(path,'--seed',str(seed),'--lr',str(info['selected_lr']),*(['--full'] if method=='full' else []))
            info['runs'][str(seed)]=str(path)
            manifest_path.write_text(json.dumps(manifest,indent=2))
    manifest['complete']=True
    manifest_path.write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()

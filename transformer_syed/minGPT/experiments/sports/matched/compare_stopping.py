"""Compare fixed epochs with a preregistered training-NLL target, using final checkpoints."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='experiments/sports/stopping')
    parser.add_argument('--epochs',type=int,default=7)
    parser.add_argument('--target-train-loss',type=float,default=0.05)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    args=parser.parse_args()
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=False)
    manifest=dict(protocol='Same shuffled epochs, initial weights and seed per pair. Final checkpoint evaluated. Loss target checked each complete epoch. No test-based selection.',
                  epochs=args.epochs,target_train_loss=args.target_train_loss,seeds=args.seeds,runs=[])
    (root/'protocol.json').write_text(json.dumps(manifest,indent=2))
    # Both learning rates were selected by the existing validation-only search.
    selection=json.loads(Path('experiments/sports/matched/selection.json').read_text())
    steps_per_epoch=(sum(1 for _ in Path('experiments/sports/train.jsonl').open())+7)//8
    for seed in args.seeds:
        for method in ['lora','full']:
            for stopping in ['epochs','train_loss']:
                out=root/f'{method}-{stopping}-seed{seed}'
                command=[sys.executable,'-u','sports_experiment.py','--out',str(out),
                         '--seed',str(seed),'--epochs',str(args.epochs),
                         '--lr',str(selection['methods'][method]['selected_lr']),
                         '--eval-every',str(steps_per_epoch),'--checkpoint-selection','final']
                if method=='full': command.append('--full')
                if stopping=='train_loss': command+=['--target-train-loss',str(args.target_train_loss)]
                subprocess.run(command,check=True)
                report=json.loads((out/'report.json').read_text())
                manifest['runs'].append(dict(method=method,stopping=stopping,seed=seed,path=str(out),
                    steps=report['completed_steps'],epochs=report['completed_epochs'],
                    stop_reason=report['stop_reason'],train_nll=report['final_training_nll'],
                    validation_nll=report['selected_validation_nll'],
                    training_step_seconds=report['training_step_seconds'],
                    metrics=report[method]['metrics']))
                (root/'summary.json').write_text(json.dumps(manifest,indent=2))
    lines=['# Epoch versus training-loss stopping','',
           f'Loss target: {args.target_train_loss}; epoch limit: {args.epochs}. Final checkpoints; same held-out sports scoring.',
           '', '| Method | Stop rule | Mean steps | Primary accuracy | Answer + scores |',
           '|---|---|---:|---:|---:|']
    for method in ['lora','full']:
        for stopping in ['epochs','train_loss']:
            rows=[r for r in manifest['runs'] if r['method']==method and r['stopping']==stopping]
            def metric(key):
                values=[100*r['metrics'][key] for r in rows]
                return f'{statistics.mean(values):.2f}% ± {statistics.stdev(values) if len(values)>1 else 0:.2f}'
            lines.append(f"| {method} | {stopping} | {statistics.mean(r['steps'] for r in rows):.0f} | {metric('primary_correct')} | {metric('fields_correct')} |")
    lines+=['','± denotes sample SD across training seeds. A run hitting the epoch cap did not reach the loss target. This is the existing synthetic benchmark.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    manifest['complete']=True
    (root/'summary.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()

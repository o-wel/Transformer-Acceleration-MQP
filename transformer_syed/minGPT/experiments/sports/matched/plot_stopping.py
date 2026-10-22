"""Save final stopping-rule comparisons and per-seed loss curves as PNG, SVG and PDF."""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('experiments/sports/stopping'))
    args=parser.parse_args()
    summary=json.loads((args.root/'summary.json').read_text())
    if not summary.get('complete'):
        raise RuntimeError('Wait for every run to complete before plotting final comparisons')
    runs=summary['runs']
    expected={(m,s,k) for m in ['lora','full'] for s in ['epochs','train_loss'] for k in summary['seeds']}
    if {(r['method'],r['stopping'],r['seed']) for r in runs}!=expected or len(runs)!=len(expected):
        raise RuntimeError('Missing or duplicated runs')
    graphs=args.root/'graphs'
    graphs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors={'epochs':'#386cb0','train_loss':'#e78b30'}
    labels={'epochs':f"{summary['epochs']} epochs",'train_loss':f"Training NLL ≤ {summary['target_train_loss']}"}
    def save(fig,name,pdf):
        fig.savefig(graphs/f'{name}.png',dpi=180,bbox_inches='tight')
        fig.savefig(graphs/f'{name}.svg',bbox_inches='tight')
        pdf.savefig(fig,bbox_inches='tight')
        plt.close(fig)
    with PdfPages(graphs/'stopping-comparison.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,key,title in zip(axes,['primary_correct','fields_correct'],['Primary-answer accuracy','Answer + score accuracy']):
            for offset,stopping in [(-0.19,'epochs'),(0.19,'train_loss')]:
                for x,method in enumerate(['lora','full']):
                    values=[100*r['metrics'][key] for r in runs if r['method']==method and r['stopping']==stopping]
                    mean=statistics.mean(values)
                    sd=statistics.stdev(values) if len(values)>1 else 0
                    ax.bar(x+offset,mean,width=.36,color=colors[stopping],yerr=sd,capsize=4,
                           label=labels[stopping] if x==0 else None)
                    ax.scatter([x+offset+(i-(len(values)-1)/2)*.05 for i in range(len(values))],values,color='#252525',s=20,zorder=3)
                    ax.text(x+offset,mean+sd+2,f'{mean:.1f}%',ha='center',fontsize=10)
            ax.set(xticks=[0,1],xticklabels=['LoRA','Full fine-tuning'],ylim=(0,110),ylabel='Held-out accuracy (%)',title=title)
            ax.legend(loc='upper right',fontsize=9)
        fig.suptitle('Final checkpoints: epoch versus training-loss stopping')
        fig.text(.5,.01,'Dots: individual seeds. Error bars: sample SD across three seeds. Same existing synthetic benchmark.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.95))
        save(fig,'accuracy',pdf)

        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,key,title,ylabel in zip(axes,['epochs','training_step_seconds'],['Training before stopping','Synchronized training compute'],['Completed epochs','Training step time (seconds)']):
            for offset,stopping in [(-.19,'epochs'),(.19,'train_loss')]:
                for x,method in enumerate(['lora','full']):
                    values=[r[key] for r in runs if r['method']==method and r['stopping']==stopping]
                    ax.bar(x+offset,statistics.mean(values),width=.36,color=colors[stopping],
                           yerr=statistics.stdev(values) if len(values)>1 else 0,capsize=4,label=labels[stopping] if x==0 else None)
                    ax.scatter([x+offset]*len(values),values,color='#252525',s=20,zorder=3)
            ax.set(xticks=[0,1],xticklabels=['LoRA','Full fine-tuning'],ylabel=ylabel,title=title)
            ax.legend(fontsize=9)
        fig.text(.5,.01,'Compute time excludes data transfer, audits, evaluation and checkpoint I/O; it is not end-to-end time.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,1))
        save(fig,'training-budget',pdf)

        fig,axes=plt.subplots(2,len(summary['seeds']),figsize=(5*len(summary['seeds']),8),squeeze=False)
        for i,method in enumerate(['lora','full']):
            for j,seed in enumerate(summary['seeds']):
                ax=axes[i,j]
                for stopping in ['epochs','train_loss']:
                    run=next(r for r in runs if r['method']==method and r['stopping']==stopping and r['seed']==seed)
                    report=json.loads((Path(run['path'])/'report.json').read_text())
                    history=report['validation_history']
                    steps_per_epoch=report['completed_steps']/report['completed_epochs']
                    ax.plot([v['step']/steps_per_epoch for v in history],[v['nll'] for v in history],
                            marker='o',color=colors[stopping],label=labels[stopping]+' (validation)')
                    if stopping=='train_loss':
                        history=report['training_nll_history']
                        ax.plot([v['step']/steps_per_epoch for v in history],[v['nll'] for v in history],
                                linestyle='--',color=colors[stopping],label='Full training-set NLL')
                ax.axhline(summary['target_train_loss'],color='gray',linestyle=':',label='Training stop threshold')
                ax.set(title=f'{method.upper()} · seed {seed}',xlabel='Epoch',ylabel='Answer NLL',yscale='log')
                ax.legend(fontsize=8)
        fig.suptitle('Loss histories and actual stopping points')
        fig.tight_layout(rect=(0,0,1,.96))
        save(fig,'loss-curves',pdf)
    (graphs/'README.md').write_text('Saved accuracy, training-budget and loss-curves graphs in PNG and SVG, plus stopping-comparison.pdf.\nSource data: ../summary.json and per-run report.json. Existing experiment graphs are preserved.\n')
    print(f'Saved graphs to {graphs}')


if __name__=='__main__':
    main()

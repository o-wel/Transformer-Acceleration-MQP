"""Build search heatmaps, Pareto fronts, combined comparisons and interaction plots."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import statistics


def dump_csv(path, rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'run_fp32')
    out=p.parse_args().out
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    trials=[json.loads(f.read_text()) for f in (out/'trials').glob('*/result.json')]
    trials=[r for r in trials if r.get('complete')]
    lines=['# Hyperparameter and combined-systems analysis','',f'Completed training trials: {len(trials)}.',
           'Selections use validation NLL; held-out test scores do not change the chosen configuration.','']
    # Screening heatmaps preserve each target/scale combination: no hidden
    # minimization across nuisance variables is presented as a marginal effect.
    screen=[r for r in trials if r['examples']==1200 and r['seed']==42]
    if screen:
        fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
        for row,targets in enumerate(['attention','all']):
            for col,scale in enumerate([1.,2.,4.]):
                values=np.full((5,3),np.nan)
                for r in screen:
                    c=r['config']
                    if c['targets']==targets and c['scale']==scale:
                        values[[2,4,8,16,32].index(c['rank']),[1e-4,3e-4,1e-3].index(c['lr'])]=r['validation_nll']
                ax=axes[row,col];im=ax.imshow(values,aspect='auto',cmap='viridis_r',
                    vmin=min(r['validation_nll'] for r in screen),vmax=max(r['validation_nll'] for r in screen))
                ax.set(title=f'{targets}; alpha/r={scale:g}',xticks=range(3),xticklabels=['1e-4','3e-4','1e-3'],
                       yticks=range(5),yticklabels=[2,4,8,16,32],xlabel='Learning rate',ylabel='Rank')
                for i,j in itertools.product(range(5),range(3)):
                    if np.isfinite(values[i,j]):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='white',fontsize=8)
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Validation NLL; 1,200-example screening budget')
        fig.savefig(out/'core_grid_heatmaps.png',dpi=180,bbox_inches='tight');plt.close(fig)
        lines += ['## Core search','',f'{len(screen)}/90 screening configurations completed.',
                  '![Rank, scaling, LR and target grid](core_grid_heatmaps.png)','']
    full=[r for r in trials if r['examples']==9600 and r['seed']==42 and r['config']['method']=='lora']
    pareto=[]
    for r in full:
        # Estimated compute-only training seconds excludes validation/checkpoint
        # overhead and includes all updates, not just the selected checkpoint.
        r['training_compute_seconds']=sum(r['step_ms'])/1000
        r['dominance_cost']=(r['validation_nll'],r['training_compute_seconds'],r['peak_training_bytes'])
    for r in full:
        if not any(all(a<=b for a,b in zip(s['dominance_cost'],r['dominance_cost'])) and
                   any(a<b for a,b in zip(s['dominance_cost'],r['dominance_cost'])) for s in full):
            pareto.append(dict(**r['config'],validation_nll=r['validation_nll'],
                               training_compute_seconds=r['training_compute_seconds'],
                               peak_training_bytes=r['peak_training_bytes'],gpu=r['gpu']))
    dump_csv(out/'PARETO.csv',pareto)
    if full:
        lines += ['## Full-budget training','',f'{len(full)} unique seed-42 trials; {len(pareto)} on the observed quality/time/memory Pareto front.',
                  'Pareto comparisons require the same GPU model. Training cost sums all update times; it excludes validation, loading and checkpoint I/O.',
                  'Detailed configurations are in TRIALS.csv; Pareto candidates are in PARETO.csv.','']
    selection=out/'selection.json'
    if selection.exists():
        sel=json.loads(selection.read_text())
        lines += ['## Frozen validation choice','','```json',json.dumps(sel['winner'],indent=2),'```','',
                  '| Candidate | Mean validation NLL | Seed standard deviation |','|---|---|---|']
        for i,r in enumerate(sel['ranked'],1):
            lines.append(f"| {i} | {r['mean_validation_nll']:.6f} | {r['stdev_validation_nll']:.6f} |")
    cells=[]
    for f in sorted(out.glob('systems-*.json')):cells+=json.loads(f.read_text())
    flattened=[]
    for cell in cells:
        row={k:v for k,v in cell.items() if k!='samples'}
        if cell['status']=='ok':
            for k in cell['samples'][0]:
                row[k+'_median']=statistics.median(s[k] for s in cell['samples'])
                row[k+'_stdev']=statistics.stdev(s[k] for s in cell['samples'])
        flattened.append(row)
    dump_csv(out/'SYSTEMS.csv',flattened)
    successful=[r for r in cells if r['status']=='ok']
    interactions=[]
    for method,batch,prompt,new in sorted({(r['method'],r['batch'],r['prompt_tokens'],r['new_tokens']) for r in successful}):
        rr=[r for r in successful if (r['method'],r['batch'],r['prompt_tokens'],r['new_tokens'])==(method,batch,prompt,new)]
        # Fix tile geometry before comparing cache interaction; selecting separate
        # fastest tiles per cache mode would conflate tuning with the interaction.
        for bm,bn,w in itertools.product([16,32],[32,64],[4,8]):
            lookup={(r['backend'],r['cache_mode']):r['wall_ms'] for r in rr
                    if r['backend']=='eager' or (r['bm'],r['bn'],r['warps'])==(bm,bn,w)}
            needed=[('eager','off'),('fa1','off'),('eager','dynamic'),('fa1','dynamic')]
            if all(k in lookup for k in needed):
                e0,f0,ec,fc=[lookup[k] for k in needed]
                interactions.append(dict(method=method,batch=batch,prompt_tokens=prompt,new_tokens=new,
                     bm=bm,bn=bn,warps=w,eager_off_ms=e0,fa1_off_ms=f0,eager_dynamic_ms=ec,fa1_dynamic_ms=fc,
                     tiled_speedup_off=e0/f0,tiled_speedup_cached=ec/fc,interaction_ratio=(e0/f0)/(ec/fc)))
    dump_csv(out/'INTERACTIONS.csv',interactions)
    if successful:
        lines += ['','## Combined systems','',f'{len(successful)} successful cells; {len(cells)-len(successful)} OOM cells.',
                  'SYSTEMS.csv includes raw-cell medians and repetition standard deviations. INTERACTIONS.csv fixes tile geometry for the cache comparison.',
                  'The fastest measured configuration below is a workload-specific performance result; correctness parity is required. It is not a quality-selection criterion.','',
                  '| Method | Batch | Prompt | Output | Fastest backend | Cache | Tiles / warps | Request ms |',
                  '|---|---|---|---|---|---|---|---|']
        keys=sorted({(r['method'],r['batch'],r['prompt_tokens'],r['new_tokens']) for r in successful})
        for key in keys:
            rr=[r for r in successful if (r['method'],r['batch'],r['prompt_tokens'],r['new_tokens'])==key]
            best=min(rr,key=lambda r:r['wall_ms'])
            lines.append(f"| {' | '.join(map(str,key))} | {best['backend']} | {best['cache_mode']} | {best['bm']}/{best['bn']}/{best['warps']} | {best['wall_ms']:.2f} |")
        # Display all policies at baseline tile geometry, avoiding best-of-eight bias.
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,method in zip(axes,['full','lora']):
            for mode,cache in itertools.product(['eager','fa1','hybrid'],['off','dynamic','static']):
                rr=sorted([r for r in successful if r['method']==method and r['batch']==1 and r['new_tokens']==64
                           and r['backend']==mode and r['cache_mode']==cache and (r['bm'],r['bn'],r['warps'])==(16,32,4)],key=lambda r:r['prompt_tokens'])
                if rr:ax.plot([r['prompt_tokens'] for r in rr],[r['wall_ms'] for r in rr],marker='o',label=f'{mode}/{cache}')
            ax.set(title=method,xlabel='Prompt tokens',ylabel='Request latency (ms)',yscale='log')
            ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(out/'combined_latency.png',dpi=180);plt.close(fig)
        lines += ['','![Combined latency at fixed tile settings](combined_latency.png)','']
    for method in ['full','lora']:
        path=out/f'{method}-test.json'
        if path.exists():
            quality=json.loads(path.read_text())
            lines += ['',f'## {method}: held-out test','', '| Metric | Mean | Seed SD |','|---|---|---|']
            for key in quality[0]['metrics']:
                values=[q['metrics'][key] for q in quality]
                sd=statistics.stdev(values) if len(values)>1 else 0
                lines.append(f'| {key} | {statistics.mean(values):.4f} | {sd:.4f} |')
            lines += [f'Completed seeds: {len(quality)}/3. Seed variability is not a confidence interval over unseen datasets.','']
    lines += ['','Hardware and budget limits: primary training and systems jobs request the academic A30 partition. Check actual GPU and software metadata before pooling runs. Full fine-tuning gets a smaller tuning budget than LoRA. Longer padded sequences do not add task context. See README.md and METHODS.md for the complete protocol.']
    (out/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

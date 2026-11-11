"""Verify completed SMS runs and create Markdown, CSV, figures, and a PDF report."""
import csv
import hashlib
import json
import statistics as stats
import textwrap
from pathlib import Path

HERE=Path(__file__).resolve().parent


def load(p):return json.loads(p.read_text())
def mean(xs):return stats.mean(xs)
def sd(xs):return stats.stdev(xs) if len(xs)>1 else 0.
def fmt(xs,percent=False):
    factor=100 if percent else 1
    return f'{factor*mean(xs):.2f} ± {factor*sd(xs):.2f}'


def main():
    complete=load(HERE/'completed.json')
    assert complete['complete'] and len(complete['runs'])==6
    selection=load(HERE/'selection.json')
    manifest=load(HERE/'data/manifest.json')
    records=[];reports={};checks=[];baseline=None
    for run in complete['runs']:
        out=Path(run['path']);r=load(out/'training.json');tests=load(out/'test.json')
        reports[(run['method'],run['seed'])]=r
        assert r['final_changes']['frozen_changed_elements']==0
        assert r['config']['lr']==selection['selected'][run['method']]['lr']
        for relative,sha in r['source_sha256'].items():
            assert hashlib.sha256((HERE.parents[1]/relative).read_bytes()).hexdigest()==sha
        assert r['data_manifest']==manifest
        for rule in ['epochs','loss_stop']:
            epoch=5 if rule=='epochs' else r['loss_stop_epoch']
            h=r['history'][epoch-1]
            t=tests[rule]
            assert t['n']==manifest['splits']['test']['n']
            pred=t['predictions']
            assert len({p['id'] for p in pred})==len(pred)
            assert sum(p['gold']==p['pred'] for p in pred)/len(pred)==t['accuracy']
            row=dict(method=run['method'],seed=run['seed'],rule=rule,lr=r['config']['lr'],epochs=epoch,
                     training_step_seconds=h['training_step_seconds'],training_wall_seconds=h['training_wall_seconds'],
                     peak_allocated_mib=h['peak_allocated_bytes']/2**20,peak_reserved_mib=h['peak_reserved_bytes']/2**20,
                     trainable_parameters=r['parameters']['trainable'],final_epoch_changed_elements=r['final_changes']['changed_elements'],
                     loss_target_reached=r['loss_target_reached'],train_nll=h['train']['label_nll'],validation_nll=h['validation']['label_nll'],
                     **{k:v for k,v in t.items() if k!='predictions'})
            records.append(row)
        if 'pretrained' in tests:baseline=tests['pretrained']
    for seed in [42,43,44]:
        a,b=(reports[(m,seed)] for m in ['lora','full'])
        assert a['initial_base_sha256']==b['initial_base_sha256']
        assert all(x['sample_order_sha256']==y['sample_order_sha256'] for x,y in zip(a['history'],b['history']))
        assert a['gpu']==b['gpu']
    assert len({r['initial_base_sha256'] for r in reports.values()})==1
    assert baseline is not None
    with (HERE/'METRICS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    groups={(m,s):[r for r in records if r['method']==m and r['rule']==s] for m in ['lora','full'] for s in ['epochs','loss_stop']}
    summary={f'{m}/{s}':{k:dict(mean=mean([r[k] for r in rr]),sd=sd([r[k] for r in rr])) for k in ['accuracy','balanced_accuracy','spam_f1','spam_precision','spam_recall','epochs','training_step_seconds','training_wall_seconds','peak_allocated_mib']} for (m,s),rr in groups.items()}
    verification=dict(passed=True,checks=['All six selected runs and both checkpoints evaluated','Identical initial base weights across runs','Identical epoch sample order for paired methods/seeds','Same GPU model across paired runs','No changes or gradients in frozen LoRA parameters','Training data/source hashes match saved reports','Test predictions reproduce recorded accuracy','Learning rates match choices written before test evaluation'])
    (HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (HERE/'summary.json').write_text(json.dumps(dict(groups=summary,pretrained={k:v for k,v in baseline.items() if k!='predictions'},data=manifest),indent=2)+'\n')
    all_training=[load(p) for p in (HERE/'runs').glob('*/training.json')]
    total_compute=sum(r['history'][-1]['training_step_seconds'] for r in all_training)
    total_wall=sum(r['total_training_wall_seconds'] for r in all_training)
    majority=manifest['splits']['test']['ham']/manifest['splits']['test']['n']
    lines=['# LoRA on a different task: real SMS spam classification','',
           'This experiment classifies messages as spam or legitimate (ham). It uses no sports examples, score reasoning, or free-text answer grading. All results below are measured on the same held-out messages.','',
           '## Data and protocol','',
           '[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), Almeida and Hidalgo (2011). '
           f"Downloaded {manifest['raw_rows']} messages; {manifest['unique_case_whitespace_messages']} remain after case/whitespace deduplication. "
           'Lowercased templates with URL/digit normalization are assigned to just one split before subsampling. This reduces common-template leakage; it does not prove all near duplicates are removed.','',
           f"- Training: {manifest['splits']['train']['n']} messages ({manifest['splits']['train']['spam']} spam).",
           f"- Validation: {manifest['splits']['validation']['n']} messages ({manifest['splits']['validation']['spam']} spam).",
           f"- Test: {manifest['splits']['test']['n']} messages ({manifest['splits']['test']['spam']} spam).",
           '- Split seed: 20261026. Training seeds: 42, 43, 44. One fixed split across seeds.',
           '- Same pretrained GPT-2, batch 8, five shuffled epochs, FP32/TF32 off, AdamW, no weight decay, gradient clipping 1.0. LoRA rank 4, alpha 8 on QKV, attention output, and MLP projections.',
           '- Equal LR search for each method: 0.001 and 0.0001, seed 42, five epochs each. Select by final validation label NLL, before test evaluation. Selected seed-42 candidates are reused.',
           '- Message content is truncated to the first 96 GPT-2 tokens, with the same instruction and label suffix for all methods. Right padding follows real tokens. Train loss is full-vocabulary next-token cross entropy for one correct label token.',
           '- Prediction chooses the higher logit of the two single-token labels ` ham` and ` spam`. This constrained classification score is used for both pretrained and trained models; it is not free-form generation.',
           '- The selected LR is shared by both stopping rules; the early-stopping rule does not receive a separate LR search.',
           '- Fixed budget: checkpoint after five epochs. Loss stopping: first epoch with full-training-set label NLL ≤ 0.05, or five-epoch cap. Both checkpoints come from the same trajectory, with no test-based selection.','',
           '## Held-out classification quality','',
           'Percentages are mean ± sample SD across three training seeds; SD is not a confidence interval. The pretrained and majority baselines are single deterministic evaluations.','',
           '| Model | Rule | Accuracy (%) | Balanced accuracy (%) | Spam F1 (%) | Spam precision (%) | Spam recall (%) |',
           '|---|---|---:|---:|---:|---:|---:|',
           f'| Always ham | No training | {100*majority:.2f} | 50.00 | 0.00 | 0.00 | 0.00 |',
           '| Pretrained GPT-2 | Two-label scoring | '+' | '.join(f'{100*baseline[k]:.2f}' for k in ['accuracy','balanced_accuracy','spam_f1','spam_precision','spam_recall'])+' |']
    for (m,s),rr in groups.items():
        lines.append(f"| {m} | {s} | "+' | '.join(fmt([r[k] for r in rr],True) for k in ['accuracy','balanced_accuracy','spam_f1','spam_precision','spam_recall'])+' |')
    lines+=['','## Training budget, time, and memory','','| Method | Rule | Epochs | Training compute (s) | Trajectory wall time (s) | Peak training allocated (MiB) |','|---|---|---:|---:|---:|---:|']
    for (m,s),rr in groups.items():
        lines.append(f'| {m} | {s} | '+' | '.join(fmt([r[k] for r in rr]) for k in ['epochs','training_step_seconds','training_wall_seconds','peak_allocated_mib'])+' |')
    lines+=['',
            'Compute time sums synchronized forward/backward/clipping/optimizer steps; it excludes transfers, audits, evaluation and checkpoint I/O. Wall time is elapsed within the training trajectory, including validation, full-training loss checks and checkpoint writes, but excludes model setup and final test/audit. Both rules share the same monitored trajectory; these are not independent end-to-end stopping-rule timing trials. Final wall time includes an earlier stopping-checkpoint write when applicable. Peak allocated GPU memory measures PyTorch allocations, not total device memory.','',
            f'Total across all {len(all_training)} training runs, including the learning-rate search: {total_compute:.1f} seconds of training compute and {total_wall:.1f} seconds summed training-loop wall time. These totals exclude model setup, final audits, and held-out testing. The comparison table reports per-run means for the selected rates.', '', '## Learning-rate selection','','| Method | Candidate LR | Final validation label NLL | Selected |','|---|---:|---:|---|']
    for c in selection['candidates']:
        lines.append(f"| {c['method']} | {c['lr']} | {c['validation_nll']:.6f} | {'yes' if c['lr']==selection['selected'][c['method']]['lr'] else 'no'} |")
    lines+=['','## What changed relative to the sports experiment?','']
    for method in ['lora','full']:
        fixed=groups[(method,'epochs')];early=groups[(method,'loss_stop')]
        acc_fixed=mean(r['accuracy'] for r in fixed);acc_early=mean(r['accuracy'] for r in early)
        saved=100*(1-mean(r['training_step_seconds'] for r in early)/mean(r['training_step_seconds'] for r in fixed))
        lines.append(f"- {method}: loss stopping changed accuracy from {100*acc_fixed:.2f}% to {100*acc_early:.2f}% ({100*(acc_early-acc_fixed):+.2f} percentage points), with {saved:.1f}% less training compute. Target reached in {sum(r['loss_target_reached'] for r in early)}/3 seeds.")
    a=mean(r['accuracy'] for r in groups[('lora','epochs')]);b=mean(r['accuracy'] for r in groups[('full','epochs')])
    lines.append(f'- Fixed epochs: LoRA/full accuracy ratio is {a/b:.3f}×; difference {100*(a-b):+.2f} percentage points. A sports-task advantage should not be assumed to transfer to this task.')
    for method in ['lora','full']:
        r=reports[(method,42)]
        lines.append(f"- {method}: {r['parameters']['trainable']:,} trainable scalars out of {r['parameters']['total']:,}; final audit found {r['final_changes']['frozen_changed_elements']:,} frozen scalars changed.")
    lines+=['',
            'A training-loss threshold controls fit to training labels, not guaranteed held-out accuracy. The numerical threshold 0.05 is reused to probe transfer of the stopping rule; one label token per example is a different objective from average loss across multi-token sports answers. Do not interpret equal numerical thresholds as equal task difficulty.','',
            'Classification asks for one of two labels, whereas the sports task required generating correct teams, scores and explanations. This experiment can show whether the observed LoRA advantage persists; it does not isolate a causal reason for any gap. Learning rates are selected separately and results depend on this dataset, split, prompt, training subset and budget.','',
            'Accuracy is inflated by class imbalance: always predicting ham already achieves the majority baseline above. Spam F1, recall and false negatives matter. The corpus is public and old; unknown GPT-2 pretraining overlap cannot be excluded. This is not evidence of performance on current SMS attacks or other classification datasets. Three seeds share a single test set, so small differences do not establish equivalence or a universal winner.','',
            '## Reproduction and artifacts','',
            'From the minGPT root, use the existing PyTorch/Transformers environment (see `../../LORA_REPRO.md` for setup). Then:',
            '```bash','python experiments/sms_spam/test_protocol.py -v',
            'sbatch experiments/sms_spam/run_gpu.sh',
            '# After all runs finish (matplotlib required):',
            'python experiments/sms_spam/build_report.py','```','',
            'The Slurm wrapper uses the author’s environment paths and cluster account; adapt them on another system. Runs refuse to overwrite the existing `runs` directory; preserve it and copy code files to a fresh sibling experiment directory for a rerun, updating the wrapper path. Dataset preparation downloads the UCI archive; GPT-2 loading uses cached weights/tokenizer. See README.md for the cache-preparation command.','',
            '- `protocol.json`: settings declared before training; `selection.json`: validation-only LR choices.',
            '- `data/manifest.json`: source checksum, split counts and SHA256 checksums.',
            '- `METRICS.csv`: each seed/rule result, confusion counts, loss, time and memory.',
            '- `runs/*/training.json`, `changes.json`, `test.json`: histories, source/initial-weight hashes, audits, and every test prediction keyed by message hash.',
            '- `verification.json`: acceptance checks; `summary.json`: aggregate results.',
            '- `REPORT.pdf`: standalone report; plots saved as PNG/SVG.',
            '- `runs/*/*.pt`: local model/adapter checkpoints; do not commit large full checkpoints.','',
            f"Hardware: {reports[('lora',42)]['gpu']}; PyTorch {reports[('lora',42)]['torch_version']}; CUDA {reports[('lora',42)]['cuda_version']}.",
            'All acceptance checks passed. The three offline tests cover scorer confusion counts, template/split separation, causal padding positions, and adapter-only training.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    figures(groups,reports,baseline,majority,manifest,selection,lines)
    print('Wrote RESULTS.md, REPORT.pdf, METRICS.csv, summary.json, verification.json',flush=True)


def figures(groups,reports,baseline,majority,manifest,selection,lines):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'epochs':'#2166ac','loss_stop':'#d6604d'}
    with PdfPages(HERE/'REPORT.pdf') as pdf:
        fig=plt.figure(figsize=(11.7,8.3))
        fig.text(.06,.92,'LoRA on a different task: SMS spam classification',fontsize=20,weight='bold')
        text=[
            'Real-message classification rather than sports question answering.',
            f"Data: {manifest['splits']['train']['n']} train / {manifest['splits']['validation']['n']} validation / {manifest['splits']['test']['n']} test; three training seeds.",
            'Duplicate/template groups stay in one split. First 96 GPT-2 message tokens; classify ham vs spam.',
            'Both methods: same GPT-2 initialization, sample order, FP32, batch 8, five epochs.',
            'LoRA rank 4 / alpha 8; compare with full fine-tuning and pretrained two-label scoring.',
            'Equal two-rate search (0.001 / 0.0001) per method; select by validation NLL before testing.',
            'Rules: final five-epoch checkpoint vs first epoch with training label NLL <= 0.05.',
            'Stopping and final checkpoints share one trajectory; no test-based checkpoint selection.',
            f'Always-ham accuracy baseline: {100*majority:.2f}%; spam F1: 0%.',
            'Read spam F1 and recall alongside accuracy because the dataset is imbalanced.',
            'Error bars: sample SD over three training seeds, all using the same fixed test set.',
            'Scope: one older public dataset, limited training subset; not a production SMS filter.',
            'Source: UCI SMS Spam Collection, Almeida and Hidalgo (2011).',
            'https://archive.ics.uci.edu/dataset/228/sms+spam+collection',
            'Raw histories, every prediction, checksums and per-seed metrics accompany this PDF.']
        y=.84
        for line in text:
            for wrapped in textwrap.wrap(line,112):
                fig.text(.06,y,wrapped,fontsize=11);y-=.030
            y-=.012
        pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11.7,5.6))
        for ax,key,title in zip(axes,['accuracy','spam_f1'],['Test accuracy','Spam F1']):
            for x,method in enumerate(['lora','full']):
                for offset,rule in [(-.18,'epochs'),(.18,'loss_stop')]:
                    values=[100*r[key] for r in groups[(method,rule)]]
                    ax.bar(x+offset,mean(values),.34,yerr=sd(values),capsize=4,color=colors[rule],label=rule if x==0 else None)
                    ax.scatter([x+offset+(i-1)*.04 for i in range(3)],values,c='black',s=15,zorder=3)
                    ax.text(x+offset,mean(values)+sd(values)+1.2,f'{mean(values):.2f}',ha='center',fontsize=9)
            ax.axhline(100*baseline[key],color='#555555',ls='--',label='pretrained')
            if key=='accuracy':ax.axhline(100*majority,color='#888888',ls=':',label='always ham')
            ax.set(xticks=[0,1],xticklabels=['LoRA','Full fine-tuning'],ylim=(0,108),ylabel='Percent',title=title)
            ax.legend(fontsize=8,loc='lower right')
        fig.suptitle('Held-out quality: dots show individual training seeds',fontsize=14)
        fig.tight_layout();fig.savefig(HERE/'quality.png',dpi=170);fig.savefig(HERE/'quality.svg');pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11.7,7.8))
        axes=axes.ravel()
        for ax,key,title in zip(axes,['epochs','training_step_seconds','training_wall_seconds','peak_allocated_mib'],['Epochs completed','Training compute (seconds)','Monitored trajectory wall (seconds)','Peak allocated GPU memory (MiB)']):
            for x,method in enumerate(['lora','full']):
                for offset,rule in [(-.18,'epochs'),(.18,'loss_stop')]:
                    values=[r[key] for r in groups[(method,rule)]]
                    ax.bar(x+offset,mean(values),.34,yerr=sd(values),capsize=4,color=colors[rule],label=rule if x==0 else None)
                    ax.text(x+offset,mean(values)*1.03,f'{mean(values):.1f}',ha='center',fontsize=8)
            ax.set(xticks=[0,1],xticklabels=['LoRA','Full'],title=title)
            ax.set_ylim(0,ax.get_ylim()[1]*1.15)
        axes[0].legend(fontsize=8)
        fig.suptitle('Stopping-rule budget and resources')
        fig.text(.5,.01,'Compute excludes evaluation/I-O. Both checkpoints share a monitored trajectory; not end-to-end job timings.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(HERE/'resources.png',dpi=170);fig.savefig(HERE/'resources.svg');pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(11.7,7.8),sharex=True)
        for i,m in enumerate(['lora','full']):
            for j,seed in enumerate([42,43,44]):
                ax=axes[i,j];r=reports[(m,seed)];h=r['history'];epochs=[x['epoch'] for x in h]
                ax.plot(epochs,[x['train']['label_nll'] for x in h],marker='o',label='Training label NLL')
                ax.plot(epochs,[x['validation']['label_nll'] for x in h],marker='s',label='Validation label NLL')
                ax.axhline(.05,c='gray',ls=':',label='Stop threshold')
                ax.axvline(r['loss_stop_epoch'],c=colors['loss_stop'],ls='--',label='Stop checkpoint')
                ax.set(title=f'{m}, seed {seed}',xlabel='Epoch',ylabel='Label NLL',yscale='log',xticks=[1,2,3,4,5])
                if i==0 and j==0:ax.legend(fontsize=7)
        fig.suptitle('Loss histories: stopping loss is not test accuracy')
        fig.tight_layout();fig.savefig(HERE/'loss-curves.png',dpi=170);fig.savefig(HERE/'loss-curves.svg');pdf.savefig(fig);plt.close(fig)
        fig=plt.figure(figsize=(11.7,8.3))
        fig.text(.06,.93,'Interpretation and limits',fontsize=20,weight='bold')
        text=[]
        for m in ['lora','full']:
            fixed=groups[(m,'epochs')];early=groups[(m,'loss_stop')]
            text.append(f"{m}: fixed accuracy {100*mean(r['accuracy'] for r in fixed):.2f}%; loss-stop accuracy {100*mean(r['accuracy'] for r in early):.2f}%. Fixed spam F1 {100*mean(r['spam_f1'] for r in fixed):.2f}%; loss-stop F1 {100*mean(r['spam_f1'] for r in early):.2f}%.")
        text += [
            'Do not assume the sports-task LoRA advantage transfers. This experiment measures a different task with the same adaptation mechanism.',
            'One-label classification and multi-token generated answers have different loss objectives. A threshold of 0.05 does not represent equal task mastery across them.',
            'Small mean differences across three seeds do not prove equivalence or a universal winner. Training seeds do not constitute independent test datasets.',
            'Prediction chooses between two label logits. The pretrained reference has this same restriction; no trained classifier head is added.',
            'Training compute excludes loss checks, validation, audits, I/O and transfers. Trajectory wall time includes checks/checkpoint writes but excludes setup and final testing. Both rules share monitoring overhead.',
            'Template grouping is heuristic; semantically similar messages may remain. The public corpus may overlap unknown pretraining data. Current SMS deployment performance is unmeasured.',
            'Acceptance checks: identical initial weights and paired epoch sample orders; same GPU; unchanged frozen LoRA weights; matching data/source hashes; all test predictions saved.',
            'See RESULTS.md and METRICS.csv for exact per-seed values, LR selection, wall times, confusion counts and reproduction steps.']
        y=.84
        for line in text:
            for wrapped in textwrap.wrap(line,112):fig.text(.06,y,wrapped,fontsize=11);y-=.025
            y-=.018
        pdf.savefig(fig);plt.close(fig)


if __name__=='__main__':main()

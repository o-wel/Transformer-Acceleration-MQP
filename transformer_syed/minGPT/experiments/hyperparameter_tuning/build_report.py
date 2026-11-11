"""Export the completed experiment as CSV, figures, and a standalone HTML report."""
import argparse
import csv
import html
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out', nargs='?', type=Path, default=Path(__file__).parent/'results')
    args = p.parse_args()
    out = args.out
    report = json.loads((out/'report.json').read_text())
    if not report['complete']:
        raise SystemExit('Experiment is still running; results are not final.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import markdown
    from experiments.hyperparameter_tuning.run_experiment import build_report
    build_report(out, report)
    comparisons = report['comparisons']
    from experiments.hyperparameter_tuning.run_experiment import BASE_LORA, BASE_CHUNK, BASE_TILE, key
    configs = {}
    for label,c,chunk,tiles in [('untuned',BASE_LORA,BASE_CHUNK,BASE_TILE),
                               ('tuned',report['selected']['lora'],report['selected']['chunk'],report['selected']['tiles'])]:
        configs[label] = dict(model='gpt2',lora=dict(enabled=True,merged=False,**c),
                              adapter_checkpoint=key(c)+'.pt',
                              kv_cache=dict(enabled=True,prefill_chunk_tokens=chunk,retain_full_context=True),
                              attention=dict(implementation='fa1_tunable',query_tile=tiles[0],key_tile=tiles[1],num_warps=tiles[2]),
                              precision='FP32 stored weights; FP16 autocast',batch=1,prompt_tokens=896,new_tokens=64)
    (out/'combined-configs.json').write_text(json.dumps(configs,indent=2))
    with (out/'COMPARISONS.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['comparison','untuned_ms','tuned_ms','speedup','bootstrap_95_low','bootstrap_95_high'])
        for name,r in comparisons.items():
            writer.writerow([name,r['baseline_ms'],r['tuned_ms'],r['speedup'],*r.get('bootstrap_95',['',''])])
    with (out/'TIMING_SAMPLES.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['configuration','repeat','wall_ms','peak_bytes'])
        for name,r in report['confirmation'].items():
            for i,s in enumerate(r['samples']):
                writer.writerow([name,i,s['wall_ms'],s['peak_bytes']])
    names = ['lora','cache','fa','combined']
    x = np.arange(len(names))
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    axes[0].bar(x-.18,[comparisons[n]['baseline_ms'] for n in names],.36,label='Untuned',color='#8895a7')
    axes[0].bar(x+.18,[comparisons[n]['tuned_ms'] for n in names],.36,label='Tuned',color='#206a83')
    axes[0].set_xticks(x,['LoRA tuning','Cache tuning','FA tuning','All three'])
    axes[0].set_ylabel('Median generation latency (ms)')
    axes[0].legend(frameon=False)
    ratios = [comparisons[n]['speedup'] for n in names]
    low = [max(0,r-comparisons[n]['bootstrap_95'][0]) for n,r in zip(names,ratios)]
    high = [max(0,comparisons[n]['bootstrap_95'][1]-r) for n,r in zip(names,ratios)]
    axes[1].bar(x,ratios,yerr=[low,high],capsize=4,color='#206a83')
    axes[1].axhline(1,color='#555',linestyle='--',linewidth=1)
    axes[1].set_xticks(x,['LoRA tuning','Cache tuning','FA tuning','All three'])
    axes[1].set_ylabel('Untuned / tuned latency (×)')
    for i,r in enumerate(ratios):
        axes[1].annotate(f'{r:.3f}×',(i,r),xytext=(0,8),textcoords='offset points',ha='center')
    fig.suptitle('Individual hyperparameter tuning and frozen combined configuration')
    fig.tight_layout()
    fig.savefig(out/'speedups.png',dpi=180)
    fig.savefig(out/'speedups.svg')
    plt.close(fig)
    text = (out/'RESULTS.md').read_text()
    text += '\n## Confirmation uncertainty\n\nPaired bootstrap 95% intervals describe timing variability within this allocation; they do not measure uncertainty across hardware or training seeds.\n\n'
    text += '| Comparison | Speedup | 95% interval |\n|---|---:|---:|\n'
    for name in names:
        r=comparisons[name]
        text += f"| {name} | {r['speedup']:.3f}× | {r['bootstrap_95'][0]:.3f}–{r['bootstrap_95'][1]:.3f}× |\n"
    text += '\n## All LoRA candidates\n\n| Rank | Alpha | LR | Validation NLL | Selected step | Training ms/step | Eligible |\n|---|---|---|---|---|---|---|\n'
    for r in report['training']:
        c=r['config']
        text += f"| {c['rank']} | {c['alpha']} | {c['lr']} | {r['validation_nll']:.5f} | {r['selected_step']} | {r['warm_median_step_ms']:.3f} | {r['validation_nll']<=report['validation_cutoff']} |\n"
    text += '\n## Runtime tuning timings\n\nThese are selection timings; final speedups above use fresh confirmation measurements.\n\n| Method | Setting | Median ms |\n|---|---|---:|\n'
    for candidate in report['tuning']['lora']:
        text += f"| lora | {candidate['config']} (eligible={candidate['eligible']}) | {candidate['median_ms']:.3f} |\n"
    for method in ('cache','fa'):
        for setting,r in report['tuning'][method].items():
            text += f"| {method} | {setting} | {r['median_ms']:.3f} |\n"
    text += '\nCache and FA sweeps and their individual confirmation comparisons hold the untuned LoRA checkpoint fixed. LoRA-only confirmation uses eager attention with caching disabled. The combined comparison changes all three selected settings together.\n'
    text += '\n## Cache numerical checks and protocol amendment\n\nThe initial runtime sweep stopped on an exact greedy-token check. Before confirmation and test evaluation, the recorded protocol was amended to require same-input logit closeness and validation NLL parity; all eight training runs were reused. FP16 execution shapes affect rounding. This amendment and the observed divergence are part of the result.\n\n| Prefill chunk | Validation NLL | Max last-prefill logit error | Greedy tokens equal full prefill |\n|---|---:|---:|---|\n'
    for chunk,r in report['tuning']['cache'].items():
        q=r['quality_check']
        text += f"| {chunk} | {q['validation_nll']:.6f} | {q['max_prefill_logit_error']:.5f} | {r['greedy_tokens_equal_full_prefill']} |\n"
    text += '\n## Held-out combined answer quality\n\n| Configuration | Primary correct | Numbers correct | Fields correct | Exact reference |\n|---|---:|---:|---:|---:|\n'
    for label in ('baseline','tuned'):
        scores=report['quality'][label]['scores']
        text += f"| {label} | {scores['primary_correct']:.3f} | {scores['score_numbers_correct']:.3f} | {scores['fields_correct']:.3f} | {scores['reference_exact']:.3f} |\n"
    text += '\n![Individual and combined speedups](speedups.png)\n' 
    (out/'RESULTS.md').write_text(text)
    import base64
    png=base64.b64encode((out/'speedups.png').read_bytes()).decode()
    body=markdown.markdown(text,extensions=['tables']).replace('src="speedups.png"',f'src="data:image/png;base64,{png}"')
    (out/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><title>Hyperparameter tuning results</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#182734}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}code{font-size:13px}</style>'+body)
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, A4
    styles = getSampleStyleSheet()
    story = []
    table_rows = []
    def flush_table():
        if not table_rows:
            return
        width = (landscape(A4)[0]-80)/len(table_rows[0])
        data = [[Paragraph(html.escape(cell), styles['BodyText']) for cell in row] for row in table_rows]
        table = Table(data, colWidths=[width]*len(table_rows[0]), repeatRows=1)
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dceaf0')),
                                   ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),
                                   ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#ccd3d9'))]))
        story.extend([table, Spacer(1,14)])
        table_rows.clear()
    for line in text.splitlines():
        if line.startswith('|'):
            cells=[c.strip() for c in line.strip('|').split('|')]
            if not all(set(c)<=set('-: ') for c in cells):
                table_rows.append(cells)
            continue
        flush_table()
        if line.startswith('!['):
            story.append(Image(str(out/'speedups.png'),width=720,height=270))
        elif line.strip():
            level=len(line)-len(line.lstrip('#'))
            style=styles['Heading1' if level==1 else 'Heading2'] if level else styles['BodyText']
            story.extend([Paragraph(html.escape(line.lstrip('# ').replace('`','')),style),Spacer(1,8)])
    flush_table()
    SimpleDocTemplate(str(out/'RESULTS.pdf'),pagesize=landscape(A4),leftMargin=40,rightMargin=40,topMargin=35,bottomMargin=35).build(story)
    print(out/'RESULTS.html')


if __name__ == '__main__':
    main()

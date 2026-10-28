"""Build a meeting-style PDF from LoRA and full-training report.json files."""
import argparse
import csv
import json
import statistics
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'experiments/lora/results'


def load_report(value, label):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if path.name != 'report.json' or not path.is_file():
        raise SystemExit(f'{label} must be an existing report.json file: {path}')
    return path, json.loads(path.read_text())


def mib(value):
    return value / 2**20 if value is not None else None


def fmt_percent(value):
    return 'N/A' if value is None else f'{value * 100:.1f}%'


def memory(report):
    return report.get('training_memory') or report.get('training_memory_through_step_40')


def page(title, paragraphs):
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(.08, .95, title, fontsize=18, weight='bold', va='top')
    y = .90
    for paragraph in paragraphs:
        wrapped = '\n'.join(textwrap.fill(line, 92) for line in str(paragraph).split('\n'))
        figure.text(.08, y, wrapped, fontsize=10, va='top', linespacing=1.45)
        y -= .021 * (wrapped.count('\n') + 1) + .018
    return figure


def quality_values(report):
    comparisons = report.get('comparisons', [])
    total = len(comparisons)
    if not total:
        return None
    base = sum(row.get('base_contains_expected', False) for row in comparisons) / total
    adapted = sum(row.get('lora_contains_expected', False) for row in comparisons) / total
    return base, adapted


def output_rows(lora_report, full_report):
    rows = []
    for row in lora_report.get('comparisons', []):
        rows.append({
            'question': row.get('question', ''),
            'reference': row.get('reference', ''),
            'pretrained': row.get('base', ''),
            'full': '',
            'lora': row.get('lora', ''),
        })
    for row in full_report.get('comparisons', []):
        match = next((item for item in rows if item['question'] == row.get('question')), None)
        if match:
            match['full'] = row.get('lora', row.get('full', ''))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lora-report', required=True,
                        help='LoRA results/report.json')
    parser.add_argument('--full-report', required=True,
                        help='Full fine-tuning results/report.json')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='Directory for LoRA_REPORT.pdf and supporting files')
    args = parser.parse_args()

    lora_path, lora = load_report(args.lora_report, 'LoRA report')
    full_path, full = load_report(args.full_report, 'Full report')
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    lp = lora.get('parameters', {})
    fp = full.get('parameters', {})
    lm = memory(lora)
    fm = memory(full)
    lora_quality = quality_values(lora)
    rows = output_rows(lora, full)
    dtype = lora.get('dtype', 'float32')
    precision = 'FP16' if '16' in dtype or lora.get('autocast') else 'FP32'
    lora_steps = lora.get('config', {}).get('steps', 'N/A')
    full_steps = full.get('config', {}).get('steps', 'N/A')

    lora_nll = lora.get('final_test_answer_nll')
    full_nll = full.get('final_test_answer_nll')
    base_nll = lora.get('base_test_answer_nll')
    lora_score = lora.get('lora_expected_phrase_accuracy')
    base_score = lora.get('base_expected_phrase_accuracy')

    summary = [
        'Research question: Can LoRA adapt the pretrained model to the held-out questions while updating fewer parameters than full fine-tuning?',
        f'Data and evaluation are taken directly from the supplied report files. This report does not retrain either model. '
        f'LoRA report: {lora_path}. Full-training report: {full_path}.',
        f'Protocol: LoRA recorded {precision}; LoRA steps {lora_steps}; full-training steps {full_steps}. '
        f'LoRA configuration: batch {lora.get("config", {}).get("batch_size", "N/A")}, '
        f'length {lora.get("config", {}).get("length", "N/A")}, rank {lora.get("config", {}).get("rank", "N/A")}, '
        f'alpha {lora.get("config", {}).get("alpha", "N/A")}.',
        f'Held-out answer NLL: pretrained base {base_nll:.4f} -> LoRA {lora_nll:.4f}; '
        f'full fine-tuning {full_nll:.4f}.' if all(x is not None for x in [base_nll, lora_nll, full_nll])
        else 'Held-out answer NLL is unavailable in one or more reports.',
        f'Expected-phrase accuracy: base {fmt_percent(base_score)} -> LoRA {fmt_percent(lora_score)}. '
        f'The supplied generic reports contain {len(rows)} fixed comparison questions.',
        f'LoRA model: {lp.get("total", 0):,} total parameters; {lp.get("trainable", 0):,} trainable '
        f'({lp.get("trainable_percent", 0):.4f}%); {lp.get("frozen", 0):,} frozen. '
        f'Full fine-tuning trainable parameters: {fp.get("trainable", 0):,}.',
        f'Storage: LoRA parameters {mib(lp.get("parameter_bytes")):.2f} MiB; '
        f'full parameters {mib(fp.get("parameter_bytes")):.2f} MiB. '
        f'Peak allocated training memory: LoRA {mib(lm.get("peak_allocated_bytes")):.2f} MiB, '
        f'full {mib(fm.get("peak_allocated_bytes")):.2f} MiB.' if lm and fm
        else 'Training-memory measurements are unavailable in one or more reports.',
        'Interpretation: this is a synthetic, format-sensitive task report. '
        'It does not establish broad conversational quality or quality equivalence between differently configured runs.',
    ]

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    quality_fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    if lora_quality:
        axes[0].bar(['Base', 'LoRA'], [lora_quality[0] * 100, lora_quality[1] * 100],
                    color=['#a5acb8', '#2867a1'])
    axes[0].set(ylim=(0, 105), ylabel='Held-out questions (%)',
                title='Expected phrase accuracy')
    nll_values = [value for value in [base_nll, full_nll, lora_nll] if value is not None]
    axes[1].bar(['Base', 'Full', 'LoRA'][:len(nll_values)], nll_values,
                color=['#a5acb8', '#cb7a27', '#2867a1'][:len(nll_values)])
    axes[1].set_ylabel('Answer NLL (nats/token)')
    axes[1].set_title('Held-out answer loss')
    quality_fig.savefig(output / 'quality.png', dpi=180)

    efficiency_fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    trainable = [fp.get('trainable', 0) / 1e6, lp.get('trainable', 0) / 1e6]
    axes[0].bar(['Full training', 'LoRA'], trainable, color=['#a5acb8', '#2867a1'])
    axes[0].set(yscale='log', ylabel='Trainable parameters (millions, log scale)',
                title='Trainable parameter count')
    for i, value in enumerate(trainable):
        axes[0].text(i, value * 1.15, f'{value:.2f}M', ha='center', fontsize=9)
    memory_values = [mib(fm.get('peak_allocated_bytes')) if fm else 0,
                     mib(lm.get('peak_allocated_bytes')) if lm else 0]
    axes[1].bar(['Full training', 'LoRA'], memory_values, color=['#a5acb8', '#2867a1'])
    axes[1].set_ylabel('Peak allocated GPU memory (MiB)')
    axes[1].set_title('Recorded training memory')
    efficiency_fig.savefig(output / 'parameters-memory.png', dpi=180)

    changes_fig, axis = plt.subplots(figsize=(9, 4.5), layout='constrained')
    history = lora.get('history', [])
    if history:
        axis.plot([item['step'] for item in history],
                  [item.get('changed_elements', 0) for item in history],
                  marker='o', label='LoRA trainable parameters')
    axis.plot([1, lora_steps if isinstance(lora_steps, int) else 1], [0, 0],
              'k--', label='Frozen base')
    axis.set(xlabel='Training step', ylabel='Scalars different from initialization',
             title='Trainable does not mean already changed')
    axis.legend(fontsize=9)
    changes_fig.savefig(output / 'changes.png', dpi=180)

    markdown = ['# LoRA versus full fine-tuning', '']
    for index, text in enumerate(summary, 1):
        markdown += [f'## {index}', text, '']
    (output / 'LoRA_REPORT.md').write_text('\n'.join(markdown))
    with (output / 'ALL_OUTPUTS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['question', 'reference', 'pretrained', 'full', 'lora'])
        writer.writeheader()
        writer.writerows(rows)

    with PdfPages(output / 'LoRA_REPORT.pdf') as pdf:
        cover = page('LoRA: unseen-match questions', summary[:5])
        pdf.savefig(cover)
        plt.close(cover)
        details = page('Parameters, memory and limits', summary[5:])
        pdf.savefig(details)
        plt.close(details)
        pdf.savefig(quality_fig)
        pdf.savefig(efficiency_fig)
        pdf.savefig(changes_fig)
        plt.close(quality_fig)
        plt.close(efficiency_fig)
        plt.close(changes_fig)
        for row in rows[:2]:
            example = page('Held-out output example', [
                row['question'], 'Reference: ' + row['reference'],
                'PRETRAINED: ' + row['pretrained'],
                'FULL: ' + row['full'], 'LORA: ' + row['lora'],
            ])
            pdf.savefig(example)
            plt.close(example)
        task = page('Task breakdown and interpretation', [
            f'Expected phrase accuracy: base {fmt_percent(base_score)}; LoRA {fmt_percent(lora_score)}.',
            f'Final answer NLL: base {base_nll if base_nll is not None else "N/A"}; '
            f'full {full_nll if full_nll is not None else "N/A"}; '
            f'LoRA {lora_nll if lora_nll is not None else "N/A"}.',
            'Raw outputs are available in ALL_OUTPUTS.csv. Review failures before making claims about quality.',
        ])
        pdf.savefig(task)
        plt.close(task)
    print(output / 'LoRA_REPORT.pdf')


if __name__ == '__main__':
    main()

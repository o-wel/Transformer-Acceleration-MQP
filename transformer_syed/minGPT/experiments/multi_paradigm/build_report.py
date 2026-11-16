"""Build a PDF report from Ax parameter-search JSON reports.

Example:
    python build_report.py --input-dir results --output results/parameter-report.pdf
"""
import argparse
import json
import math
import platform
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


BYTES_PER_MIB = 1024 ** 2
OBJECTIVE_DEFINITIONS = (
    ("training_time", "Training time (s)", "lower"),
    ("training_memory", "Training peak memory (MiB)", "lower"),
    ("validation_loss", "Validation loss", "lower"),
    ("validation_fields_correct", "Validation fields correct (%)", "higher"),
    ("inference_time", "Inference time (ms)", "lower"),
)


def finite(value):
    return value is not None and math.isfinite(float(value))


def load_reports(input_dir, output_path):
    reports = []
    for path in sorted(input_dir.rglob("*.json")):
        if path.resolve() == output_path.resolve():
            continue
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not read {path}: {error}") from error
        if isinstance(report, dict) and "trials" in report and "search" in report:
            reports.append((path, report))
    if not reports:
        raise ValueError(f"No parameter-search reports found under {input_dir}")
    return reports


def completed_trials(report):
    return [
        trial for trial in report.get("trials", [])
        if trial.get("status") == "completed" and trial.get("objectives")
    ]


def metric_value(trial, name):
    value = trial.get("objectives", {}).get(name)
    if value is None:
        benchmark = trial.get("benchmark", {})
        training = benchmark.get("training", {})
        inference = benchmark.get("inference", {})
        validation = benchmark.get("validation_inference", {})
        value = {
            "training_time": training.get("seconds"),
            "training_memory": training.get("peak_memory_allocated_bytes"),
            "validation_loss": training.get("final_validation_loss"),
            "validation_fields_correct": validation.get("scores", {}).get("fields_correct"),
            "inference_time": inference.get("wall_ms", {}).get("mean"),
        }.get(name)
    if not finite(value):
        return None
    value = float(value)
    return value / BYTES_PER_MIB if name == "training_memory" else (
        value * 100 if name == "validation_fields_correct" else value
    )


def report_objectives(report):
    configured = report.get("search", {}).get("objectives", {})
    return [
        definition for definition in OBJECTIVE_DEFINITIONS
        if definition[0] in configured
    ]


def best_trial(report, trials):
    objectives = report_objectives(report)
    if not objectives:
        return None
    valid = [trial for trial in trials if all(
        metric_value(trial, name) is not None for name, _, _ in objectives
    )]
    if not valid:
        return None
    # Normalize each metric by the observed range, then average the four
    # normalized costs. This gives a reproducible summary without hiding the
    # original objective values shown elsewhere in the report.
    ranges = {}
    for name, _, direction in objectives:
        values = [metric_value(trial, name) for trial in valid]
        low, high = min(values), max(values)
        ranges[name] = (low, high, direction)

    def score(trial):
        costs = []
        for name, _, direction in objectives:
            low, high, direction = ranges[name]
            value = metric_value(trial, name)
            normalized = 0.0 if high == low else (value - low) / (high - low)
            costs.append(normalized if direction == "lower" else 1 - normalized)
        return sum(costs) / len(costs)

    return min(valid, key=score)


def label(report_path, report):
    optimization = report.get("search", {}).get("optimizations", {})
    enabled = [name for name in ("lora", "kvcache", "flash_attention")
               if optimization.get(name)]
    return report.get("report_name", report_path.stem) + (
        f" ({', '.join(enabled)})" if enabled else ""
    )


def trial_page(report_path, report):
    trials = completed_trials(report)
    title = label(report_path, report)
    if not trials:
        figure = plt.figure(figsize=(11.69, 8.27))
        figure.text(.06, .92, title, fontsize=18, weight="bold")
        figure.text(.06, .82, "No completed trials were found.", fontsize=12)
        return figure

    figure, axes = plt.subplots(2, 4, figsize=(14, 8.27))
    objectives = report_objectives(report)
    for ax, (name, ylabel, direction) in zip(axes.flat, objectives):
        selected = [(trial, metric_value(trial, name)) for trial in trials]
        selected = [(trial, value) for trial, value in selected if value is not None]
        x = list(range(len(selected)))
        y = [value for _, value in selected]
        ax.plot(x, y, "o-", color="#2867a1", linewidth=1.2, markersize=4)
        if y:
            best_index = (min if direction == "lower" else max)(range(len(y)), key=y.__getitem__)
            ax.scatter([best_index], [y[best_index]], color="#d55e00", zorder=3, s=55)
        ax.set_title(f"{ylabel} ({direction} is better)")
        ax.set_xlabel("Trial order")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.25)
    for ax in axes.flat[len(objectives):]:
        ax.axis("off")
    parameter_specs = (
        ("learning_rate", "validation_loss", "Learning rate vs validation loss",
         "Learning rate", "Validation loss"),
        ("batch_size", "training_time", "Batch size vs training time",
         "Batch size", "Training time (s)"),
        ("length", "training_memory", "Length vs training memory",
         "Sequence length", "Training memory (MiB)"),
    )
    for ax, (parameter, metric, plot_title, xlabel, ylabel) in zip(
        axes.flat[len(objectives):], parameter_specs
    ):
        ax.axis("on")
        points = [
            (trial.get("parameters", {}).get(parameter), metric_value(trial, metric))
            for trial in trials
        ]
        points = [(x, y) for x, y in points if finite(x) and y is not None]
        if points:
            ax.scatter([x for x, _ in points], [y for _, y in points],
                       color="#2867a1", s=32)
        ax.set_title(plot_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
    figure.suptitle(f"{title}: hyperparameter tuning ({len(trials)} completed trials)",
                    fontsize=16, y=.98)
    figure.subplots_adjust(top=.86, bottom=.12, left=.06, right=.98,
                           hspace=.42, wspace=.32)
    return figure


def trial_hyperparameter_page(report_path, report):
    trials = completed_trials(report)
    title = label(report_path, report)
    chosen = best_trial(report, trials)
    parameter_names = ["learning_rate", "batch_size", "length"]
    for trial in trials:
        for name in trial.get("parameters", {}):
            if name not in parameter_names:
                parameter_names.append(name)
    columns = ["trial_index"] + parameter_names
    headings = ["Trial"] + [name.replace("_", " ").title() for name in parameter_names]

    def format_value(value):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.5g}"
        return str(value)

    body = [
        [format_value(trial.get("trial_index"))] + [
            format_value(trial.get("parameters", {}).get(name))
            for name in parameter_names
        ]
        for trial in trials
    ]
    figure, axis = plt.subplots(figsize=(14, 8.27))
    axis.axis("off")
    axis.set_title(f"{title}: trial hyperparameters", loc="left",
                   fontsize=16, weight="bold", pad=18)
    if not body:
        axis.text(.02, .82, "No completed trials were found.", fontsize=12)
        return figure
    table = axis.table(cellText=body, colLabels=headings, loc="upper center",
                       cellLoc="center", bbox=[.02, .08, .96, .76])
    table.auto_set_font_size(False)
    table.set_fontsize(9 if len(body) <= 25 else 7)
    table.auto_set_column_width(list(range(len(headings))))
    best_index = chosen.get("trial_index") if chosen else None
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d8dee8")
        if row_index == 0:
            cell.set_facecolor("#e6eff4")
            cell.set_text_props(weight="bold")
        elif row_index - 1 < len(trials):
            trial_index = trials[row_index - 1].get("trial_index")
            if trial_index == best_index:
                cell.set_facecolor("#fff0b3")
                cell.set_text_props(weight="bold")
    axis.text(.02, .02, "Highlighted row: selected best trial based on this report's objectives.",
              fontsize=9, color="#555555")
    return figure


def comparison_rows(reports):
    rows = []
    for path, report in reports:
        trials = completed_trials(report)
        chosen = best_trial(report, trials)
        # KV-cache searches do not optimize training metrics, but their
        # benchmark still records them. Keep a completed trial available for
        # the training comparison graphs when objective selection is absent.
        if chosen is None and report.get("search", {}).get(
            "optimizations", {}).get("kvcache") and trials:
            chosen = trials[0]
        if chosen is None:
            continue
        benchmark = chosen.get("benchmark", {})
        training = benchmark.get("training", {})
        validation = benchmark.get("validation_inference", {})
        inference = benchmark.get("inference", {})
        params = chosen.get("parameters", {})
        rows.append({
            "configuration": label(path, report),
            "trial": chosen.get("trial_index"),
            "learning_rate": params.get("learning_rate"),
            "batch_size": params.get("batch_size"),
            "length": params.get("length"),
            "rank": params.get("rank"),
            "alpha": params.get("alpha"),
            "training_time": training.get("seconds", metric_value(chosen, "training_time")),
            "training_memory": training.get(
                "peak_memory_allocated_bytes",
                metric_value(chosen, "training_memory") * BYTES_PER_MIB
                if metric_value(chosen, "training_memory") is not None else None,
            ),
            "validation_loss": training.get("final_validation_loss"),
            "validation_accuracy": validation.get("scores", {}).get("fields_correct"),
            "test_loss": training.get("final_test_loss"),
            "test_accuracy": inference.get("scores", {}).get("fields_correct"),
            "inference_time": inference.get("wall_ms", {}).get("mean"),
            "inference_memory": inference.get("peak_memory_allocated_bytes"),
        })
    return rows


def comparison_figure(rows, key, title, ylabel, scale=1, percent=False):
    selected = [row for row in rows if finite(row.get(key))]
    figure, axis = plt.subplots(figsize=(11.69, 5.2), layout="constrained")
    values = [float(row[key]) * scale for row in selected]
    axis.bar(range(len(selected)), values, color="#2867a1")
    labels = [textwrap.fill(row["configuration"], 22) for row in selected]
    axis.set(title=title, ylabel=ylabel, xticks=range(len(selected)),
             xticklabels=labels)
    axis.tick_params(axis="x", labelrotation=25, labelsize=8)
    for index, value in enumerate(values):
        text = f"{value:.1f}%" if percent else f"{value:.2f}"
        axis.text(index, value, text, ha="center", va="bottom", fontsize=8)
    return figure


def table_page(rows):
    columns = [
        ("configuration", "Configuration"),
        ("trial", "Trial"),
        ("learning_rate", "LR"),
        ("batch_size", "Batch"),
        ("length", "Length"),
        ("rank", "Rank"),
        ("alpha", "Alpha"),
        ("training_time", "Train s"),
        ("training_memory", "Train MiB"),
        ("inference_time", "Infer ms"),
        ("validation_loss", "Val loss"),
        ("validation_accuracy", "Val fields"),
        ("test_accuracy", "Test fields"),
    ]
    def display(key, value):
        if value is None:
            return "-"
        if key == "training_memory":
            return f"{float(value) / BYTES_PER_MIB:.0f}"
        if key in {"learning_rate", "validation_loss"}:
            return f"{float(value):.4g}"
        if key in {"training_time", "inference_time"}:
            return f"{float(value):.1f}"
        if key in {"validation_accuracy", "test_accuracy"}:
            return f"{float(value) * 100:.1f}%"
        return str(value)

    body = []
    for row in rows:
        body.append([
            textwrap.fill(display(key, row[key]), 22)
            for key, _ in columns
        ])
    figure, axis = plt.subplots(figsize=(14, 8.27), layout="constrained")
    axis.axis("off")
    axis.set_title("Best trial selected from each parameter-search report", loc="left",
                   fontsize=16, weight="bold", pad=12)
    table = axis.table(cellText=body, colLabels=[name for _, name in columns],
                       loc="upper center", cellLoc="center", bbox=[0, .38, 1, .48])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(list(range(len(columns))))
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d8dee8")
        if row_index == 0:
            cell.set_facecolor("#e6eff4")
            cell.set_text_props(weight="bold")

    axis.text(0, .32, "Rows are the trials Ax selected as Pareto-optimal; no trial was selected by this report builder.",
              fontsize=9, color="#555555")
    return figure


def environment_page(reports):
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.text(.06, .94, "Search environments", fontsize=18, weight="bold")
    y = .87
    for path, report in reports:
        environment = report.get("environment", {})
        text = (
            f"{label(path, report)}\n"
            f"Python {environment.get('python', '-')}; PyTorch {environment.get('torch', '-')}; "
            f"CUDA {environment.get('cuda', '-')}; GPU {environment.get('gpu', '-')}\n"
            f"Platform: {environment.get('platform', '-')}; executable: {environment.get('executable', '-')}"
        )
        wrapped = "\n".join(textwrap.fill(line, 125) for line in text.splitlines())
        figure.text(.06, y, wrapped, fontsize=10, va="top", linespacing=1.4)
        y -= .13
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("parameter-report.pdf"))
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output = args.output if args.output.is_absolute() else input_dir / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = load_reports(input_dir, output)
    rows = comparison_rows(reports)

    with PdfPages(output) as pdf:
        pdf.savefig(environment_page(reports))
        plt.close()
        for path, report in reports:
            figure = trial_page(path, report)
            pdf.savefig(figure)
            plt.close(figure)
            figure = trial_hyperparameter_page(path, report)
            pdf.savefig(figure)
            plt.close(figure)
        if rows:
            figure = table_page(rows)
            pdf.savefig(figure)
            plt.close(figure)
            comparison_specs = [
                ("training_time", "Training time comparison", "Training time (s)", 1, False),
                ("inference_time", "Inference time comparison", "Mean inference time (ms)", 1, False),
                ("training_memory", "Training memory comparison", "Peak training memory (MiB)",
                 1 / BYTES_PER_MIB, False),
                ("inference_memory", "Inference memory comparison", "Peak inference memory (MiB)",
                 1 / BYTES_PER_MIB, False),
                ("validation_loss", "Validation loss comparison", "Final validation loss", 1, False),
                ("test_loss", "Test loss comparison", "Final test loss", 1, False),
                ("validation_accuracy", "Validation accuracy comparison",
                 "Validation fields correct (%)", 100, True),
                ("test_accuracy", "Test accuracy comparison",
                 "Test fields correct (%)", 100, True),
            ]
            for key, title, ylabel, scale, percent in comparison_specs:
                figure = comparison_figure(rows, key, title, ylabel, scale, percent)
                pdf.savefig(figure)
                plt.close(figure)
    summary = {
        "input_directory": str(input_dir),
        "output": str(output),
        "report_count": len(reports),
        "completed_report_count": len(rows),
        "reports": [
            {"file": str(path), "report_name": label(path, report),
             "completed_trials": len(completed_trials(report))}
            for path, report in reports
        ],
        "best_trials": rows,
        "builder": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"Processed {len(reports)} search reports and {len(rows)} best trials")
    print(f"Saved PDF to {output}")


if __name__ == "__main__":
    main()

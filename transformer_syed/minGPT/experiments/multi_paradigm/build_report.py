"""Build comparison reports from multi-paradigm benchmark JSON files.

Example:
    python build_report.py --input-dir results --output-dir results/comparison
"""
import argparse
import csv
import json
import math
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


DEFAULTS = {
    "device": "cuda",
    "dtype": "float32",
    "steps": 300,
    "batch_size": 4,
    "length": 128,
    "new_tokens": 40,
    "lr": 0.001,
    "clip_norm": 1.0,
    "rank": 4,
    "alpha": 8.0,
    "inference_repeats": 3,
    "seed": 42,
    "threads": 4,
    "lora": False,
    "kvcache": False,
    "flash": False,
}


def finite(value):
    return value is not None and math.isfinite(float(value))


def mean(values):
    values = [float(value) for value in values if finite(value)]
    return sum(values) / len(values) if values else None


def maximum(values):
    values = [int(value) for value in values if value is not None]
    return max(values) if values else None


def load_reports(input_dir, output_dir):
    reports = []
    for path in sorted(input_dir.rglob("*.json")):
        if (output_dir == path.parent or output_dir in path.parents
                or path.name == "comparison-report.json"):
            continue
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not read report {path}: {error}") from error
        if not isinstance(report, dict) or "training" not in report or "inference" not in report:
            raise ValueError(f"{path} is not a multi-paradigm benchmark report")
        reports.append((path, report))
    if not reports:
        raise ValueError(f"No benchmark JSON reports found under {input_dir}")
    return reports


def hyperparameters(report):
    config = report.get("config", {})
    enhancements = report.get("enhancements", {})
    values = {name: config.get(name, enhancements.get(name, default))
              for name, default in DEFAULTS.items()}
    return values


def label_for(report, path):
    values = hyperparameters(report)
    changed = [
        name.replace("_", "-") if isinstance(value, bool) else f"{name.replace('_', '-')}-{value}"
        for name, default in DEFAULTS.items()
        if (value := values[name]) != default
    ]
    return path.stem if not changed else f"{path.stem} [{', '.join(changed)}]"


def summarize(path, report):
    training = report["training"]
    inference = report["inference"]
    training_history = training.get("history", [])
    inference_steps = inference.get("memory_steps", [])
    inference_wall = inference.get("wall_ms", {})
    scores = inference.get("scores", {})
    return {
        "file": str(path),
        "name": path.stem,
        "label": label_for(report, path),
        **hyperparameters(report),
        "training_seconds": training.get("seconds"),
        "training_tokens_per_second": training.get("tokens_per_second"),
        "initial_test_loss": training.get("initial_test_loss"),
        "final_test_loss": training.get("final_test_loss"),
        "training_mean_step_ms": mean(row.get("wall_ms") for row in training_history),
        "training_peak_allocated_bytes": maximum(
            row.get("memory", {}).get("peak_allocated_bytes") for row in training_history
        ) or training.get("peak_memory_allocated_bytes"),
        "training_mean_allocated_bytes": mean(
            row.get("memory", {}).get("allocated_bytes") for row in training_history
        ),
        "inference_mean_wall_ms": inference_wall.get("mean"),
        "inference_tokens_per_second": inference.get("tokens_per_second"),
        "inference_primary_correct": scores.get("primary_correct"),
        "inference_fields_correct": scores.get("fields_correct"),
        "inference_reference_exact": scores.get("reference_exact"),
        "inference_peak_allocated_bytes": maximum(
            step.get("memory", {}).get("peak_allocated_bytes") for step in inference_steps
        ) or inference.get("peak_memory_allocated_bytes"),
        "inference_mean_allocated_bytes": mean(
            step.get("memory", {}).get("allocated_bytes") for step in inference_steps
        ),
    }


def values(rows, key):
    return [row[key] for row in rows if finite(row.get(key))]


def plot_metric(rows, key, title, ylabel, output, formatter=None):
    labels = [row["label"] for row in rows]
    numbers = [float(row[key]) / (1024 ** 2) if "bytes" in key else float(row[key])
               for row in rows if finite(row.get(key))]
    selected = [row for row in rows if finite(row.get(key))]
    labels = [row["label"] for row in selected]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5), layout="constrained")
    ax.bar(range(len(labels)), numbers, color="#2867a1")
    ax.set(title=title, ylabel=ylabel, xticks=range(len(labels)), xticklabels=labels)
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    for index, value in enumerate(numbers):
        text = formatter(value) if formatter else f"{value:.2f}"
        ax.text(index, value, text, ha="center", va="bottom", fontsize=8)
    fig.savefig(output, dpi=180)
    return fig


def build_figures(rows, output_dir):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    figures = []
    metrics = [
        ("training_seconds", "Training runtime by configuration", "Seconds"),
        ("inference_mean_wall_ms", "Inference runtime by configuration", "Mean inference pass (ms)"),
        ("training_peak_allocated_bytes", "Peak training memory by configuration", "Peak allocated CUDA memory (MiB)"),
        ("inference_peak_allocated_bytes", "Peak inference memory by configuration", "Peak allocated CUDA memory (MiB)"),
        ("inference_fields_correct", "Inference answer correctness", "Correct answers with required fields (%)"),
        ("final_test_loss", "Final held-out test loss", "Cross-entropy loss"),
    ]
    for key, title, ylabel in metrics:
        formatter = (lambda value: f"{value:.1f}%") if "correct" in key else None
        if "correct" in key:
            # Plot percentages instead of fractions.
            rows = [dict(row, **{key: float(row[key]) * 100}) if finite(row.get(key)) else row
                    for row in rows]
        figure = plot_metric(rows, key, title, ylabel,
                             output_dir / f"{key}.png", formatter=formatter)
        figures.append((f"{key}.png", title, figure))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    panels = [
        ("training_seconds", "Training runtime (s)"),
        ("inference_mean_wall_ms", "Inference runtime (ms)"),
        ("training_peak_allocated_bytes", "Training peak memory (MiB)"),
        ("inference_peak_allocated_bytes", "Inference peak memory (MiB)"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        selected = [row for row in rows if finite(row.get(key))]
        numbers = [float(row[key]) / (1024 ** 2) if "bytes" in key else float(row[key])
                   for row in selected]
        ax.bar(range(len(selected)), numbers, color="#2867a1")
        ax.set(title=title, xticks=range(len(selected)),
               xticklabels=[row["label"] for row in selected])
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
    fig.savefig(output_dir / "runtime-memory-overview.png", dpi=180)
    figures.append(("runtime-memory-overview.png", "Runtime and memory overview", fig))
    return figures


def write_csv(rows, path):
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results"),
                        help="Directory containing benchmark JSON files; subdirectories are included")
    parser.add_argument("--output-dir", type=Path, default=Path("report"),
                        help="Directory for the consolidated report and figures")
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "comparison").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = load_reports(input_dir, output_dir)
    rows = [summarize(path, report) for path, report in reports]
    write_csv(rows, output_dir / "summary.csv")
    figures = build_figures(rows, output_dir)
    with PdfPages(output_dir / "comparison-report.pdf") as pdf:
        for _, title, figure in figures:
            figure.suptitle(title, fontsize=15)
            pdf.savefig(figure)
            plt.close(figure)
    consolidated = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "report_count": len(rows),
        "reports": rows,
        "figures": [filename for filename, _, _ in figures],
        "builder": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
    }
    (output_dir / "comparison-report.json").write_text(json.dumps(consolidated, indent=2))
    print(f"Processed {len(rows)} reports")
    print(f"Saved comparison report to {output_dir}")


if __name__ == "__main__":
    main()
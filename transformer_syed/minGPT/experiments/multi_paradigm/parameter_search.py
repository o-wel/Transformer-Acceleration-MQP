"""Use AxClient to search benchmark hyperparameters.

The benchmark is run once per Ax trial. Trial reports are suppressed with
``--no-report``; this script stores the compact benchmark result in one
``parameter_report.json`` file.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
from ax.service.ax_client import AxClient, ObjectiveProperties

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = Path(__file__).with_name("benchmark.py")


def report_path_with_optimizations(path, args):
    enabled = [
        name for name, enabled in (
            ("lora", args.lora),
            ("kvcache", args.kvcache),
            ("flash", args.flash),
        )
        if enabled
    ]
    if not enabled:
        return path
    return path.with_name(path.stem + "_" + "_".join(enabled) + path.suffix)


def run_trial(parameters, args, trial_index):
    command = [
        sys.executable, str(BENCHMARK),
        "--train", str(args.train),
        "--validation", str(args.validation),
        "--test", str(args.test),
        "--device", args.device,
        "--dtype", args.dtype,
        "--steps", str(args.steps),
        "--new-tokens", str(args.new_tokens),
        "--lr", str(parameters["learning_rate"]),
        "--batch-size", str(parameters["batch_size"]),
        "--length", str(parameters["length"]),
        "--clip-norm", str(args.clip_norm),
        "--rank", str(parameters.get("rank", args.rank)),
        "--alpha", str(parameters.get("alpha", args.alpha)),
        "--inference-repeats", str(args.inference_repeats),
        "--seed", str(args.seed + trial_index),
        "--threads", str(args.threads),
        "--no-report",
    ]
    if args.lora:
        command.append("--lora")
    if args.kvcache:
        command.append("--kvcache")
    if args.flash:
        command.append("--flash")
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    elapsed = time.perf_counter() - started
    marker = "PARAMETER_SEARCH_RESULT="
    result_line = next(
        (line for line in reversed(process.stdout.splitlines()) if line.startswith(marker)),
        None,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Trial {trial_index} failed with exit code {process.returncode}:\n"
            f"{process.stderr[-4000:] or process.stdout[-4000:]}"
        )
    if result_line is None:
        raise RuntimeError(f"Trial {trial_index} did not return a compact benchmark result")
    return json.loads(result_line[len(marker):]), elapsed


def objective_values(report, include_training_objectives):
    training = report["training"]
    validation = report["validation_inference"]
    inference = report["inference"]
    objectives = {
        "validation_loss": float(training["final_validation_loss"]),
        "validation_fields_correct": float(validation["scores"]["fields_correct"]),
        "inference_time": float(inference["wall_ms"]["mean"]),
    }
    if include_training_objectives:
        training = report["training"]
        objectives.update({
            "training_time": float(training["seconds"]),
            "training_memory": float(training["peak_memory_allocated_bytes"]),
        })
    return objectives


def serialize_pareto_parameters(pareto_parameters):
    return {
        str(trial_index): {
            "parameters": parameters,
            "objectives": means,
            "covariance": covariance,
        }
        for trial_index, (parameters, (means, covariance))
        in pareto_parameters.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--report", type=Path,
                        default=Path("experiments/multi_paradigm/results/parameter_report.json"))
    parser.add_argument("--train", default="experiments/sports/train.jsonl")
    parser.add_argument("--validation", default="experiments/sports/validation.jsonl")
    parser.add_argument("--test", default="experiments/sports/test.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--new-tokens", type=int, default=40)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--inference-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--kvcache", "--kv-cache", dest="kvcache", action="store_true")
    parser.add_argument("--flash", "--flash-attention", dest="flash", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.steps < 1 or args.timeout < 1:
        parser.error("--trials, --steps, and --timeout must be positive")
    if args.device != "cuda":
        parser.error("The memory objective requires --device cuda")
    if args.flash and args.dtype == "float32":
        backend = "tiled_fp32"
    elif args.flash:
        backend = "flash"
    else:
        backend = "eager"
    kv_cache_only = args.kvcache and not args.lora and not args.flash
    kv_cache_only = False
    objectives = {
        "validation_loss": ObjectiveProperties(minimize=True),
        "validation_fields_correct": ObjectiveProperties(minimize=False),
        "inference_time": ObjectiveProperties(minimize=True),
    }
    if not kv_cache_only:
        objectives = {
            "training_time": ObjectiveProperties(minimize=True),
            "training_memory": ObjectiveProperties(minimize=True),
            **objectives,
        }

    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    report_path = report_path_with_optimizations(report_path, args)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ax_client = AxClient(random_seed=args.seed)
    search_parameters = [
        {
            "name": "learning_rate",
            "type": "range",
            "bounds": [1e-5, 3e-3],
            "log_scale": True,
        },
        {"name": "batch_size", "type": "choice", "values": [2, 4, 8, 16]},
        {"name": "length", "type": "choice", "values": [128, 256, 512]},
    ]
    if args.lora:
        search_parameters.extend([
            {"name": "rank", "type": "choice", "values": [2, 4, 8, 16, 32]},
            {"name": "alpha", "type": "choice", "values": [4.0, 8.0, 16.0, 32.0, 64.0]},
        ])
    ax_client.create_experiment(
        name="minGPT_parameter_search",
        parameters=search_parameters,
        objectives=objectives,
    )
    search_report = {
        "complete": False,
        "report_name": report_path.stem,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": str(ROOT),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
        "search": {
            "trials_requested": args.trials,
            "benchmark": str(BENCHMARK),
            "backend": backend,
            "optimizations": {
                "lora": args.lora,
                "kvcache": args.kvcache,
                "flash_attention": args.flash,
            },
            "fixed_parameters": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "search_parameters": {
                "learning_rate": {"bounds": [1e-5, 3e-3], "log_scale": True},
                "batch_size": [2, 4, 8, 16],
                "length": [128, 256, 512],
                **({
                    "rank": [2, 4, 8, 16, 32],
                    "alpha": [4.0, 8.0, 16.0, 32.0, 64.0],
                } if args.lora else {}),
            },
            "objectives": {
                name: "maximize" if properties.minimize is False else "minimize"
                for name, properties in objectives.items()
            },
        },
        "trials": [],
    }

    for trial_index in range(args.trials):
        parameters, trial_index = ax_client.get_next_trial()
        entry = {"trial_index": trial_index, "parameters": parameters, "status": "running"}
        try:
            compact_report, wall_time = run_trial(parameters, args, trial_index)
            metrics = objective_values(compact_report, include_training_objectives=not kv_cache_only)
            ax_client.complete_trial(trial_index=trial_index, raw_data=metrics)
            entry.update({
                "status": "completed",
                "wall_time": wall_time,
                "objectives": metrics,
                "benchmark": compact_report,
            })
        except Exception as error:
            ax_client.log_trial_failure(trial_index=trial_index)
            entry.update({"status": "failed", "error": str(error)})
        search_report["trials"].append(entry)
        report_path.write_text(json.dumps(search_report, indent=2))
        print(json.dumps(entry, separators=(",", ":")), flush=True)

    search_report["complete"] = all(
        trial["status"] == "completed" for trial in search_report["trials"]
    )
    pareto_parameters = ax_client.get_pareto_optimal_parameters()
    search_report["ax_pareto_optimal_parameters"] = {
        "selection": "Ax model-predicted Pareto frontier",
        "trials": serialize_pareto_parameters(pareto_parameters),
    }
    search_report["ax_client"] = ax_client.to_json_snapshot()
    report_path.write_text(json.dumps(search_report, indent=2))
    print(f"Saved parameter search report to {report_path}")


if __name__ == "__main__":
    main()
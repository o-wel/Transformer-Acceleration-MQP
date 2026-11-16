#!/usr/bin/env bash
#SBATCH --account=isp
#SBATCH --partition=academic
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=mqp-fp16-tuning
set -eo pipefail
source "$HOME/MQP/Torch/setup.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0
export TMPDIR="$MQP_RUN/tmp/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$MQP_RUN/triton-cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
export PYTHONPATH="${MQP_REPORT_DEPS}"
if [[ -n "${MQP_SNAPSHOT:-}" ]]; then
    python "$MQP_FROZEN/integrity.py" "$MQP_SNAPSHOT"
fi
python -u "$MQP_FROZEN/study.py" "$1" --out "$MQP_RUN" --task "$MQP_TASK" --precision "$MQP_PRECISION" --shard "${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$1" == report ]]; then
    python "$MQP_FROZEN/analyze.py" --out "$MQP_RUN"
    python "$MQP_FROZEN/build_report.py" --out "$MQP_RUN"
    python "$MQP_FROZEN/compare.py" "$MQP_SNAPSHOT"
fi

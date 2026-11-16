#!/usr/bin/env bash
#SBATCH --account=isp
#SBATCH --partition=academic
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=mqp-opt-matrix
set -eo pipefail
source "$HOME/MQP/Torch/setup.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0
export TMPDIR="$MQP_RUN/tmp/$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$MQP_RUN/triton-cache/$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
python -u "$MQP_MATRIX/study.py" "$1" --out "$MQP_RUN" --shard "${SLURM_ARRAY_TASK_ID:-0}"

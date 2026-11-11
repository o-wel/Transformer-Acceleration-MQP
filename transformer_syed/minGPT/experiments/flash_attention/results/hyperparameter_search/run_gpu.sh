#!/usr/bin/env bash
#SBATCH --account=isp
#SBATCH --partition=academic
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=mqp-hyperparams
set -eo pipefail
source "$HOME/MQP/Torch/setup.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0
study_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
# Slurm copies the script to its spool directory; submit.sh exports the real path.
study_dir="${MQP_SEARCH_DIR:-$study_dir}"
python -u "$study_dir/study.py" "$1" --out "$MQP_SEARCH_OUT" --shard "${SLURM_ARRAY_TASK_ID:-0}"

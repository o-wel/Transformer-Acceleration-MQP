#!/usr/bin/env bash
#SBATCH --account=yli15
#SBATCH --partition=quick
#SBATCH --qos=quick
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:08:00
#SBATCH --job-name=mqp-fp32-a100
#SBATCH --output=experiments/fp32_attention/a100-%j.log
set -eo pipefail
source "$HOME/MQP/Torch/setup.sh"
cd "$HOME/MQP/MinGPT/minGPT"
export NVIDIA_TF32_OVERRIDE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -u experiments/fp32_attention/a100_followup.py

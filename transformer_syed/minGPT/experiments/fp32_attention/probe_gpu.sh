#!/usr/bin/env bash
#SBATCH --account=yli15
#SBATCH --partition=quick
#SBATCH --qos=quick
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=mqp-fp32-probe
#SBATCH --output=experiments/fp32_attention/probe-%j.log
set -eo pipefail
source "$HOME/MQP/Torch/setup.sh"
cd "$HOME/MQP/MinGPT/minGPT"
export NVIDIA_TF32_OVERRIDE=0
python -u experiments/fp32_attention/probe.py

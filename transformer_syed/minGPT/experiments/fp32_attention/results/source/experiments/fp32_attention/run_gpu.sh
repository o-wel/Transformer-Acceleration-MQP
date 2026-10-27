#!/usr/bin/env bash
#SBATCH --account=isp
#SBATCH --partition=academic
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=mqp-fp32-attn
#SBATCH --output=experiments/fp32_attention/gpu-%j.log
set -eo pipefail
source "$HOME/MQP/Torch/setup.sh"
cd "$HOME/MQP/MinGPT/minGPT"
export NVIDIA_TF32_OVERRIDE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -u experiments/fp32_attention/run_experiment.py
python experiments/fp32_attention/audit_kernels.py

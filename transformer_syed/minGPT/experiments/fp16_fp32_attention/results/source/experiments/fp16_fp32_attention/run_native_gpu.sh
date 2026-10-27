#!/usr/bin/env bash
#SBATCH --account=isp
#SBATCH --partition=academic
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:15:00
#SBATCH --job-name=mqp-native-fp16
#SBATCH --output=experiments/fp16_fp32_attention/native-%j.log
set -eo pipefail
source "$HOME/MQP/Torch/setup.sh"
cd "$HOME/MQP/MinGPT/minGPT"
export NVIDIA_TF32_OVERRIDE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -u experiments/fp16_fp32_attention/native_inference.py

#!/usr/bin/env bash
#SBATCH --account=isp
#SBATCH --partition=academic
#SBATCH --qos=short
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:08:00
#SBATCH --job-name=mqp-joint-quality
#SBATCH --output=experiments/joint_decode/results/quality-gpu-%j.log
set -eo pipefail
source "$HOME/MQP/Torch/setup.sh"
cd "$HOME/MQP/MinGPT/minGPT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -u -m experiments.joint_decode.quality

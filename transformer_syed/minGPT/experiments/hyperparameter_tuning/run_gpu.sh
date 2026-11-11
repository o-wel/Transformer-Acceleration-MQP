#!/usr/bin/env bash
#SBATCH --account=isp
#SBATCH --partition=academic
#SBATCH --qos=short
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=mqp-hyperparams
#SBATCH --output=/home/sashazli/MQP/MinGPT/minGPT/experiments/hyperparameter_tuning/gpu-%j.log
set -eo pipefail
source "$HOME/MQP/Torch/setup.sh"
cd "$HOME/MQP/MinGPT/minGPT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NVIDIA_TF32_OVERRIDE=0
python -u experiments/hyperparameter_tuning/run_experiment.py "$@"

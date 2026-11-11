#!/usr/bin/env bash
#SBATCH --account=isp
#SBATCH --partition=academic
#SBATCH --qos=short
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --job-name=mqp-graph-confirm
#SBATCH --output=experiments/graph_decode/results/confirmation-%j.log
set -eo pipefail
source "$HOME/MQP/Torch/setup.sh"
cd "$HOME/MQP/MinGPT/minGPT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -u -m experiments.graph_decode.confirm "$@"

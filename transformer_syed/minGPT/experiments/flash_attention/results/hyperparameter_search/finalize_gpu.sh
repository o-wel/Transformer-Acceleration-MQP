#!/usr/bin/env bash
#SBATCH --account=isp
#SBATCH --partition=academic
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=mqp-search-report
set -eo pipefail
source "$HOME/MQP/Torch/setup.sh"
python "$MQP_SEARCH_DIR/analyze.py" --out "$MQP_SEARCH_OUT"

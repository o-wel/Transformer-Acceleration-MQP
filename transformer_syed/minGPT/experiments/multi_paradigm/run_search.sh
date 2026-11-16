#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time 0-12:00:00 
#SBATCH --job-name=mqp-param-search
#SBATCH --constraint="A100"
#SBATCH --output=gpu-%j.log
set -eo pipefail
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate MQP
cd "$HOME/MQP/minGPT/experiments/multi_paradigm"

python parameter_search.py
python parameter_search.py --lora
python parameter_search.py --kvcache
python parameter_search.py --flash
python parameter_search.py --lora --kvcache --flash
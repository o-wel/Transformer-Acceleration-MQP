#!/usr/bin/env bash
#SBATCH --partition=quick
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=mqp-lora
#SBATCH --output=experiments/lora/gpu-%j.log
set -eo pipefail
source "$HOME/MQP/Torch/setup.sh"
cd "$HOME/MQP/MinGPT/minGPT"
python -u lora_experiment.py train --out out/lora-sports-run1
python -u lora_experiment.py train --full --steps 10 --out out/full-sports-run1
python experiments/lora/summarize.py

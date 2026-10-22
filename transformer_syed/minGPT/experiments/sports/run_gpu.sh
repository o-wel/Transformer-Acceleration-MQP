#!/usr/bin/env bash
#SBATCH --account=isp
#SBATCH --partition=academic
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --job-name=mqp-sports
#SBATCH --output=experiments/sports/gpu-%j.log
set -eo pipefail
source "$HOME/MQP/Torch/setup.sh"
cd "$HOME/MQP/MinGPT/minGPT"
for seed in 42 43 44; do
    python -u sports_experiment.py --out "out/sports-seed${seed}" --seed "$seed"
done
python -u sports_experiment.py --out out/sports-full-memory --full --steps 40 --seed 42

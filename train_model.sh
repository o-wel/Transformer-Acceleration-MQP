#!/bin/bash
#SBATCH -N 1        
#SBATCH -n 2       
#SBATCH --mem=8g    
#SBATCH --job-name "Transformer Training"
#SBATCH --partition short    
#SBATCH --time 0-12:00:00 
#SBATCH --gres=gpu:2
#SBATCH --constraint="A100"
#SBATCH --output jobOut-%j.out

conda activate MQP

python LoRA_transformer_owen/train.py

conda deactivate
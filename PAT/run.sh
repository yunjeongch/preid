#!/bin/bash

#SBATCH --job-name=both      
#SBATCH --partition=a5000        
#SBATCH --gres=gpu:1        
#SBATCH --time=5-12:00:00
#SBATCH --mem=100000
#SBATCH --cpus-per-task=16
#SBATCH --output=./out/market_both.log   

ml purge
ml load cuda/11.8
eval "$(conda shell.bash hook)"  # Initialize Conda Environment
conda activate pat             # Activate your conda environment

python train.py --config_file "config/PAT.yml"
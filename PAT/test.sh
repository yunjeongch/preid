#!/bin/bash

#SBATCH --job-name=cuhk       
#SBATCH --partition=a5000        
#SBATCH --gres=gpu:1        
#SBATCH --time=5-12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=./out/test/market_cuhk.log   

ml purge
ml load cuda/11.8
eval "$(conda shell.bash hook)"  # Initialize Conda Environment
conda activate pat             # Activate your conda environment

config_file='config/PAT_org.yml'

python test.py --config_file $config_file #--pose True #--save True ## set --save if saving wrong samples
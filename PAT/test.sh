#!/bin/bash

#SBATCH --job-name=pat_msmt       
#SBATCH --partition=a6000        
#SBATCH --gres=gpu:1        
#SBATCH --time=5-12:00:00
#SBATCH --mem=100000
#SBATCH --cpus-per-task=8
#SBATCH --output=./msmt_market_inf.log   

ml purge
ml load cuda/11.8
eval "$(conda shell.bash hook)"  # Initialize Conda Environment
conda activate pat             # Activate your conda environment

config_file='config/PAT.yml'

python test.py --config_file $config_file --save True ## set --save if saving wrong samples
#!/bin/bash

#SBATCH --job-name=pat_vis       
#SBATCH --partition=a6000        
#SBATCH --gres=gpu:0        
#SBATCH --time=5-12:00:00
#SBATCH --mem=100000
#SBATCH --cpus-per-task=8
#SBATCH --output=./market_msmt_vis.log   

ml purge
ml load cuda/11.8
eval "$(conda shell.bash hook)"  # Initialize Conda Environment
conda activate pat             # Activate your conda environment

pred_path='inf/market_msmt'
dset_name='market_msmt' ## 

python visualize.py --pred_path $pred_path --dset_name $dset_name
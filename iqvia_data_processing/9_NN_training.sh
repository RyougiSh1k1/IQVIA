#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=9_NN_training
#SBATCH --output=9_NN_training.out
#SBATCH --error=9_NN_training.err

# Load modules
module load anaconda3/current
module load cuda11.8/toolkit/11.8.0

# Initialize conda (if needed)
source /cm/shared/apps/anaconda3/current/etc/profile.d/conda.sh

# Activate environment
conda activate iqvia_env

# Or if conda still doesn't work, just ensure tensorflow is installed
python -m pip install tensorflow

# Run the script
python 9_NN_training.py > 9_NN_training.log
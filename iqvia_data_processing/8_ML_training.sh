#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=8_ML_training
#SBATCH --output=8_ML_training.out
#SBATCH --error=8_ML_training.err

# Load the anaconda3 module
module load anaconda3/current

# Run the script
srun python 8_ML_training.py > 8_ML_training.log
#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=9_NN_training
#SBATCH --output=9_NN_training.out
#SBATCH --error=9_NN_training.err

# Load the anaconda3 module
module load anaconda3/current

# Run the script
srun python 9_NN_training.py > 9_NN_training.log
#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=5_merge_features
#SBATCH --output=5_merge_features.out
#SBATCH --error=5_merge_features.err

# Load the anaconda3 module
module load anaconda3/current

# Run the script
srun python 5_merge_features.py > 5_merge_features.log
#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=7_merge_OUD
#SBATCH --output=7_merge_OUD.out
#SBATCH --error=7_merge_OUD.err

# Load the anaconda3 module
module load anaconda3/current

# Run the script
srun python 7_merge_OUD.py > 7_merge_OUD.log
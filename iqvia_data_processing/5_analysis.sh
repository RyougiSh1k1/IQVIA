#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=5_analysis
#SBATCH --output=5_analysis.out
#SBATCH --error=5_analysis.err

# Load the anaconda3 module
module load anaconda3/current

# Run the script
srun python 5_analysis.py > 5_analysis.log
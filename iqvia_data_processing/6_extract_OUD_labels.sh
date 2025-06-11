#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=6_extract_OUD_labels
#SBATCH --output=6_extract_OUD_labels.out
#SBATCH --error=6_extract_OUD_labels.err

# Load the anaconda3 module
module load anaconda3/current

# Run the script
srun python 6_extract_OUD_labels.py > 6_extract_OUD_labels.log
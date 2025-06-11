#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=find_claims_files
#SBATCH --output=find_claims_files.out
#SBATCH --error=find_claims_files.err

# Load the anaconda3 module
module load anaconda3/current

# Run the script
srun python find_claims_files.py > find_claims_files.log
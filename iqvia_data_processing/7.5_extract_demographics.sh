#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=7.5_extract_demographics
#SBATCH --output=7.5_extract_demographics.out
#SBATCH --error=7.5_extract_demographics.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Load the anaconda3 module
module load anaconda3/current

# Run the script
echo "Starting demographic feature extraction..."
echo "This process will extract age, gender, zip3, and payment type"
echo "Processing enrollment synthetic data and enrollment files from 2006-2022"

srun python 7.5_extract_demographics.py > logs/7.5_extract_demographics.log 2>&1

echo "Process completed. Check the log file for details."
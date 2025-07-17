#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=9.1_simple_NN
#SBATCH --output=9.1_simple_NN.out
#SBATCH --error=9.1_simple_NN.err

# Load modules
module load anaconda3/current
module load cuda11.8/toolkit/11.8.0

# Set CUDA environment variables
export CUDA_HOME=/cm/shared/apps/cuda11.8/toolkit/11.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Also add the CUDA stubs directory (important for TensorFlow)
export LD_LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$LD_LIBRARY_PATH

# Set TensorFlow to use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Verify CUDA installation
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Checking for libcudart.so:"
ls -la $CUDA_HOME/lib64/libcudart.so*

# Show GPU info
nvidia-smi

# Run the script
python 9.1_simple_NN.py > 9.1_simple_NN.log
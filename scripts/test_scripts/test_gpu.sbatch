#!/bin/bash
#SBATCH --job-name=gpu-check
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=gpu-check-%j.out
#SBATCH --error=gpu-check-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Activate your python virtual environment if you created one
# source asl_env/bin/activate

python3 check_gpu.py
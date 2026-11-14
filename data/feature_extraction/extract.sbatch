#!/bin/bash
# Extract keypoints for every clip in manifest.csv, split across 5 CPU nodes.
#   Submit:        sbatch data/feature_extraction/extract.sbatch
#   Quick test:    sbatch --array=0 --export=ALL,LIMIT=50 data/feature_extraction/extract.sbatch
#   Watch:         squeue -u $USER ; tail -f /hpc/gkobea/asl-data/logs/extract_*.out
# Safe to resubmit: finished clips are skipped.
#SBATCH --job-name=asl-extract
#SBATCH --partition=students
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=03:30:00
#SBATCH --output=/hpc/gkobea/asl-data/logs/extract_%A_%a.out

set -euo pipefail
source /hpc/gkobea/miniconda3/etc/profile.d/conda.sh
conda activate asl

NUM_SHARDS=${NUM_SHARDS:-5}
LIMIT=${LIMIT:-0}
cd /hpc/gkobea/asl_project
echo "Node $(hostname), shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}, $(nproc) CPUs"

python data/feature_extraction/extract_keypoints.py \
    --data-root /hpc/gkobea/asl-data \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${NUM_SHARDS}" \
    --workers "${SLURM_CPUS_PER_TASK}" --limit "${LIMIT}"

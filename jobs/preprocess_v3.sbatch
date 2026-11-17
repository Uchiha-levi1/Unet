#!/bin/bash
# Full preprocessing, sharded over a job array (one shard per task). Use jobs/preprocess_v3_submit.sh.
#SBATCH -J prep_v3
#SBATCH -p short
#SBATCH --account=nitin
#SBATCH --array=0-3
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o %x_%A_%a.out

set -euo pipefail
REPO=/home/vchezhian/scratch/Unet
SRC=/scratch/mhameed/shared_dataset/single_a22_physical_v13
cd "$REPO"

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate "$REPO/env"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m src.preprocess_v3 --src "$SRC" --out Data_v3 --shard "$SLURM_ARRAY_TASK_ID" --nshards "$SLURM_ARRAY_TASK_COUNT"

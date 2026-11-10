#!/bin/bash
# Full compositing, sharded over a job array (one shard per task). Use jobs/composite_v2_submit.sh.
#SBATCH -J comp_v2
#SBATCH -p short
#SBATCH --account=nitin
#SBATCH --array=0-3
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o %x_%A_%a.out

set -euo pipefail
REPO=/home/vchezhian/scratch/Unet
FG=/scratch/mhameed/shared_dataset/foregrounds
cd "$REPO"

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate "$REPO/env"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS=(--fg "$FG" --v1 Data_v1 --bg bg --out Data_v2)

python -m src.composite_v2 "${ARGS[@]}" --shard "$SLURM_ARRAY_TASK_ID" --nshards "$SLURM_ARRAY_TASK_COUNT"

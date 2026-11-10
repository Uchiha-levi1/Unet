#!/bin/bash
# Builds Data_v1/index.csv, qa.csv, defect_states.json, stats.json after the array. Use jobs/preprocess_v1_submit.sh.
#SBATCH -J prep_v1_final
#SBATCH -p short
#SBATCH --account=nitin
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
REPO=/home/vchezhian/scratch/Unet
SRC=/scratch/mhameed/shared_dataset/composites/variant_001
cd "$REPO"

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate "$REPO/env"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m src.preprocess_v1 --src "$SRC" --out Data_v1 --finalize

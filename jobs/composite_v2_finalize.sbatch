#!/bin/bash
# Builds Data_v2/index.csv and stats.json after the array. Use jobs/composite_v2_submit.sh.
#SBATCH -J comp_v2_final
#SBATCH -p short
#SBATCH --account=nitin
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
REPO=/home/vchezhian/scratch/Unet
FG=/scratch/mhameed/shared_dataset/foregrounds
cd "$REPO"

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate "$REPO/env"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS=(--fg "$FG" --v1 Data_v1 --bg bg --out Data_v2)

python -m src.composite_v2 "${ARGS[@]}" --finalize

#!/bin/bash
# Test preprocessing on 2 inspections (npz + PNGs) into Data_v1/. Submit from the repo root:
#   sbatch jobs/preprocess_test.sbatch
#SBATCH -J prep_v1_test
#SBATCH -p short
#SBATCH --account=nitin
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o %x_%j.out

set -euo pipefail
REPO=/home/vchezhian/scratch/Unet
SRC=/scratch/mhameed/shared_dataset/composites/variant_001
cd "$REPO"

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate "$REPO/env"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m src.preprocess_v1 --src "$SRC" --out Data_v1 --png --only family_000001_baseline family_000001_E03
python -m src.preprocess_v1 --src "$SRC" --out Data_v1 --finalize

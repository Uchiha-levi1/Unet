#!/bin/bash
# Test preprocessing on 2 families (train f000000, test f000850) (npz + PNGs) into Data_v3/. Submit from the repo root:
#   sbatch jobs/preprocess_test.sbatch
#SBATCH -J prep_v3_test
#SBATCH -p short
#SBATCH --account=nitin
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o %x_%j.out

set -euo pipefail
REPO=/home/vchezhian/scratch/Unet
SRC=/scratch/mhameed/shared_dataset/single_a22_physical_v13
cd "$REPO"

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate "$REPO/env"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m src.preprocess_v3 --src "$SRC" --out Data_v3 --png --only f000000 f000850
python -m src.preprocess_v3 --src "$SRC" --out Data_v3 --finalize

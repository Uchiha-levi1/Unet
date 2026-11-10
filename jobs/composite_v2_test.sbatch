#!/bin/bash
# Test compositing on 2 inspections (npz + PNGs) into Data_v2/. Submit from the repo root:
#   sbatch jobs/composite_v2_test.sbatch
#SBATCH -J comp_v2_test
#SBATCH -p short
#SBATCH --account=nitin
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 00:30:00
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

python -m src.composite_v2 "${ARGS[@]}" --png --only family_000001_baseline family_000001_E03
python -m src.composite_v2 "${ARGS[@]}" --finalize

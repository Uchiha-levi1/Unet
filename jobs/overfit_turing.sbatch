#!/bin/bash
# Overfit sanity check (4 full 768x768 views, ~200 steps) on Turing. Submit from the repo root:
#   sbatch jobs/overfit_turing.sbatch
#SBATCH -J unet3p_overfit
#SBATCH -p short
#SBATCH --account=nitin
#SBATCH --gres=gpu:L40S:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
REPO=/home/vchezhian/scratch/Unet
cd "$REPO"

module load miniconda3            # exact module name: check with `module avail miniconda`
eval "$(conda shell.bash hook)"
conda activate "$REPO/env"
export HF_HOME=/home/vchezhian/scratch/hf_cache HF_HUB_OFFLINE=1

python -m tests.overfit --config configs/turing.yaml --plot runs/overfit_loss.png

#!/bin/bash
# UNet 3+ training on Turing (WPI). Submit from the repo root:   sbatch jobs/train_turing.sbatch
# Other config (e.g. the ablation):                               CONFIG=configs/no_cldice.yaml sbatch jobs/train_turing.sbatch
# Resubmitting the same command resumes from the newest epoch_XXX.pt snapshot in the config's out_dir.
#SBATCH -J unet3p_a22
#SBATCH -p short
#SBATCH --account=nitin
#SBATCH --gres=gpu:L40S:1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
REPO=/home/vchezhian/scratch/Unet
CONFIG=${CONFIG:-configs/turing.yaml}
cd "$REPO"

module load miniconda3            # exact module name: check with `module avail miniconda`
eval "$(conda shell.bash hook)"
conda activate "$REPO/env"
export HF_HOME=/home/vchezhian/scratch/hf_cache HF_HUB_OFFLINE=1   # pretrained weights fetched once on the login node

# Stage the training data on node-local /tmp (fast, wiped when the job ends). Checkpoints/logs stay in $REPO/runs.
STAGE=/tmp/$SLURM_JOB_ID/Data
mkdir -p "$STAGE"
trap 'rm -rf /tmp/$SLURM_JOB_ID' EXIT
cp -r "$REPO/Data/npz" "$REPO/Data/index.csv" "$REPO/Data/stats.json" "$STAGE/"
echo "staged $(du -sh "$STAGE" | cut -f1) to $STAGE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NGPU" -m src.train --config "$CONFIG" --data-root "$STAGE"
else
    python -m src.train --config "$CONFIG" --data-root "$STAGE"
fi

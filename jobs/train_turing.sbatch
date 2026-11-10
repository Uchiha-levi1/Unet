#!/bin/bash
# UNet 3+ training on Turing (WPI). Submit from the repo root:   sbatch jobs/train_turing.sbatch
# Other config (e.g. the ablation):                               CONFIG=configs/no_cldice.yaml sbatch jobs/train_turing.sbatch
# Run 2 fine-tune (Data_v1):
#   CONFIG=configs/v1_finetune.yaml DATA=Data_v1 INIT=runs/turing/best.pt sbatch -t 12:00:00 jobs/train_turing.sbatch
# Resubmitting the same command resumes from the newest epoch_XXX.pt snapshot (INIT is then ignored).
#SBATCH -J unet3p_a22
#SBATCH -p short
#SBATCH --account=nitin
#SBATCH --gres=gpu:L40S:1
#SBATCH -c 10
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
REPO=/home/vchezhian/scratch/Unet
CONFIG=${CONFIG:-configs/turing.yaml}
DATA=${DATA:-Data}                 # dataset folder in $REPO
INIT=${INIT:-}                     # optional: initial weights (fine-tuning)
cd "$REPO"

module load miniconda3            # exact module name: check with `module avail miniconda`
eval "$(conda shell.bash hook)"
conda activate "$REPO/env"
export HF_HOME=/home/vchezhian/scratch/hf_cache HF_HUB_OFFLINE=1   # pretrained weights fetched once on the login node

# Stage the data on node-local /tmp (fast, wiped when the job ends) if it has room for the data + 10 GB;
# otherwise read directly from scratch. Checkpoints/logs stay in $REPO/runs.
NEED=$(( $(du -sb "$REPO/$DATA/npz" | cut -f1) + 10 * 1024**3 ))
FREE=$(df --output=avail -B1 /tmp | tail -1)
echo "data $(( NEED / 1024**3 - 10 )) GB, /tmp free $(( FREE / 1024**3 )) GB"
if [ "$FREE" -ge "$NEED" ]; then
    ROOT=/tmp/$SLURM_JOB_ID/$DATA
    mkdir -p "$ROOT"
    trap 'rm -rf /tmp/$SLURM_JOB_ID' EXIT
    T0=$SECONDS
    cp -r "$REPO/$DATA/npz" "$REPO/$DATA/index.csv" "$REPO/$DATA/stats.json" "$ROOT/"
    echo "staged $(du -sh "$ROOT" | cut -f1) to $ROOT in $(( SECONDS - T0 ))s"
else
    ROOT=$REPO/$DATA
    echo "not enough /tmp space: reading directly from $ROOT"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ARGS=(--config "$CONFIG" --data-root "$ROOT")
[ -n "$INIT" ] && ARGS+=(--init-checkpoint "$INIT")
NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NGPU" -m src.train "${ARGS[@]}"
else
    python -m src.train "${ARGS[@]}"
fi

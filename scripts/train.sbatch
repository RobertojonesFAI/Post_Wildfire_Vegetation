#!/bin/bash
# Train one model on a GPU node. Run from the repo folder on Borah, with DATASET exported:
#   mkdir -p logs
#   sbatch scripts/train.sbatch <run name> [more train.py options]
# e.g.
#   sbatch scripts/train.sbatch frozen_300m --backbone prithvi_eo_v2_300_tl
#   sbatch scripts/train.sbatch full_300m --backbone prithvi_eo_v2_300_tl --unfreeze --lr 1e-4
#
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Add "#SBATCH --partition=<name>" for the GPU partition you want (list them with: sinfo)

set -euo pipefail
NAME="$1"
shift

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python src/recovery/train.py --dataset "$DATASET" --out "runs/$NAME" --workers "$SLURM_CPUS_PER_TASK" "$@"

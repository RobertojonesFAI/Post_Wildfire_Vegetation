#!/bin/bash
# Step 3: build the training dataset from the exported chips.
# Run from the repo folder on Borah:
#   mkdir -p logs
#   sbatch scripts/build_dataset.sbatch /path/to/cs497_hls_chips /path/to/dataset
#
#SBATCH --job-name=build_dataset
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# If Borah asks for a partition, add a line like "#SBATCH --partition=<name>" (list them with: sinfo)

set -euo pipefail
CHIPS="$1"
OUT="$2"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python scripts/build_dataset.py --chips "$CHIPS" --out "$OUT" --workers "$SLURM_CPUS_PER_TASK"

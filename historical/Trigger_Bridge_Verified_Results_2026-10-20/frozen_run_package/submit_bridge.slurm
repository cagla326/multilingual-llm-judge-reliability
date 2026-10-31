#!/bin/bash
#SBATCH --job-name=trigger_bridge
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=trigger_bridge_%j.out
#SBATCH --error=trigger_bridge_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from bridge package folder}"
module load cuda/12.4.1
: "${EXTENSION_PYTHON:?Export the verified interpreter}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$EXTENSION_PYTHON" -u run_bridge.py "${1:?smoke or infer}" --out "${2:?run directory}"

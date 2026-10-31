#!/bin/bash
#SBATCH --job-name=ko_verifier
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=ko_verifier_%j.out
#SBATCH --error=ko_verifier_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the package folder}"
module load cuda/12.4.1
: "${EXTENSION_PYTHON:?Export the verified Python interpreter before sbatch}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$EXTENSION_PYTHON" -u verifier.py "${1:?smoke, train or assess}" --out "${2:?run directory}"

#!/bin/bash
#SBATCH --job-name=ko_klue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=ko_klue_%j.out
#SBATCH --error=ko_klue_%j.err
set -euo pipefail
module load cuda/12.4.1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
extension_python="${EXTENSION_PYTHON:?Export the verified Python interpreter before sbatch}"
cd "${SLURM_SUBMIT_DIR}"
if [ "$1" = "all" ]; then
  "$extension_python" -u run_klue.py evaluate --out "$2"
  "$extension_python" -u run_klue.py ablation --out "$2"
  echo "COMPLETE all"
else
  "$extension_python" -u run_klue.py "$1" --out "$2"
fi

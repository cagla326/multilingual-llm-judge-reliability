#!/bin/bash
#SBATCH --job-name=ko_extension
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=ko_extension_%j.out
#SBATCH --error=ko_extension_%j.err
set -euo pipefail
module load cuda/12.4.1
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
extension_python="${EXTENSION_PYTHON:-python3}"
extension_root="${SLURM_SUBMIT_DIR}"
extension_task="${1:?Use smoke, robustness, or transfer}"
extension_out="${2:?Supply the absolute frozen-run output directory}"
cd "$extension_root"
"$extension_python" -u run_extension.py "$extension_task" --out "$extension_out"

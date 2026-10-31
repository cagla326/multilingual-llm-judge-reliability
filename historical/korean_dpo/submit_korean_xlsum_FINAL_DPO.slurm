#!/bin/bash
#SBATCH --job-name=ko_xlsum_final_dpo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=ko_xlsum_final_dpo_%j.out
#SBATCH --error=ko_xlsum_final_dpo_%j.err

set -euo pipefail

cd "$HOME/korean_training/lora_experiment"

module load cuda/12.4.1

export TMPDIR=/tmp
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "============================================================"
echo "FINAL KOREAN XL-SUM DPO"
echo "============================================================"
date
hostname
nvidia-smi

echo
echo "Verifying frozen inputs..."
sha256sum -c korean_xlsum_TRAIN50_LOCK_SHA256.txt
sha256sum -c FINAL_DPO_FREEZE_SHA256.txt

echo
echo "Starting final training..."
python3 -u train_korean_xlsum_FINAL_DPO.py

echo
echo "TRAINING FINISHED"
date

echo
echo "Adapter:"
ls -lh gemma3_4b_ko_xlsum_FINAL_DPO_train100

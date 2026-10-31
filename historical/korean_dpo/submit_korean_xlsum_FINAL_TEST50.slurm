#!/bin/bash
#SBATCH --job-name=ko_final_test50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=ko_final_test50_%j.out
#SBATCH --error=ko_final_test50_%j.err

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
echo "FINAL KOREAN XL-SUM FROZEN TEST50"
echo "============================================================"

date
hostname
nvidia-smi

echo
echo "===== VERIFY FROZEN EVALUATION FILES ====="

sha256sum -c FINAL_TEST50_EVAL_FREEZE_SHA256.txt

echo
echo "===== CHECK OUTPUT DIRECTORY ====="

if [ -e FINAL_KOREAN_TEST50_EVAL ]; then
    echo "ERROR: FINAL_KOREAN_TEST50_EVAL already exists."
    echo "Refusing to overwrite final evaluation."
    exit 1
fi

echo "Output path clean: OK"

echo
echo "===== START FINAL EVALUATION ====="

python3 -u eval_korean_xlsum_FINAL_TEST50.py

echo
echo "===== FREEZE FINAL OUTPUTS ====="

sha256sum FINAL_KOREAN_TEST50_EVAL/* \
    > FINAL_KOREAN_TEST50_EVAL_OUTPUT_SHA256.txt

chmod 444 FINAL_KOREAN_TEST50_EVAL/*
chmod 555 FINAL_KOREAN_TEST50_EVAL
chmod 444 FINAL_KOREAN_TEST50_EVAL_OUTPUT_SHA256.txt

echo
echo "FINAL TEST50 EVALUATION FINISHED"
date

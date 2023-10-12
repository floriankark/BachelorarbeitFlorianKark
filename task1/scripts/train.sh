#!/bin/bash
#PBS -l select=1:ncpus=20:mem=50gb:ngpus=1:accelerator_model=a100
#PBS -l walltime=00:59:00
#PBS -A "MM_ClaimWorth"
 
set -e
 
module load Miniconda/3.1

conda activate my_env

# Example for RoBERTa-large with max token length of 128 on the text and OCR data
python /gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/src/text_model.py \
    --tokenizer_max_len 128 \
    --epochs 20 \
    --learning_rate 2e-5 \
    --model "roberta-large" \
    --ocr True

conda deactivate
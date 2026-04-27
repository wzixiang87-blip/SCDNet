#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

#!/bin/bash
model_name=SCDNet
root_path=${ROOT_PATH:-./dataset/electricity/}
data_path=${DATA_PATH:-electricity.csv}

echo "========== Electricity: pred_len = 96 =========="
python -u run.py \
  --is_training 1 \
  --root_path /root/autodl-tmp/all_datasets/electricity \
  --data_path electricity.csv \
  --model_id "ECL_96_96" \
  --model SCDNet \
  --data custom \
  --features M \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 321 \
  --des Exp \
  --d_model 512 \
  --d_ff 1024 \
  --n_heads 4 \
  --e_layers 1 \
  --dropout 0.1 \
  --output_proj_dropout 0.1 \
  --batch_size 16 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate 0.0005 \
  --lradj type1 \
  --cycle 168 \
  --n_clusters 32 \
  --top_k 4 \
  --itr 1

echo "========== Electricity: pred_len = 192 =========="
python -u run.py \
  --is_training 1 \
    --root_path /root/autodl-tmp/all_datasets/electricity \
  --data_path electricity.csv \
  --model_id "ECL_96_192" \
  --model SCDNet \
  --data custom \
  --features M \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 321 \
  --des Exp \
  --d_model 512 \
  --d_ff 1024 \
  --n_heads 4 \
  --e_layers 1 \
  --dropout 0.1 \
  --output_proj_dropout 0.1 \
  --batch_size 16 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate 0.0005 \
  --lradj type1 \
  --cycle 168 \
  --n_clusters 32 \
  --top_k 4 \
  --itr 1

echo "========== Electricity: pred_len = 336 =========="
python -u run.py \
  --is_training 1 \
   --root_path /root/autodl-tmp/all_datasets/electricity \
  --data_path electricity.csv \
  --model_id "ECL_96_336" \
  --model SCDNet \
  --data custom \
  --features M \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 321 \
  --des Exp \
  --d_model 512 \
  --d_ff 1024 \
  --n_heads 4 \
  --e_layers 2 \
  --dropout 0.1 \
  --output_proj_dropout 0.1 \
  --batch_size 16 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate 0.0005 \
  --lradj type1 \
  --cycle 168 \
  --n_clusters 32 \
  --top_k 4 \
  --itr 1

echo "========== Electricity: pred_len = 720 =========="
python -u run.py \
  --is_training 1 \
   --root_path /root/autodl-tmp/all_datasets/electricity \
  --data_path electricity.csv \
  --model_id "ECL_96_720" \
  --model SCDNet \
  --data custom \
  --features M \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 321 \
  --des Exp \
  --d_model 512 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 2 \
  --dropout 0.1 \
  --output_proj_dropout 0.1 \
  --batch_size 16 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate 0.0005 \
  --lradj type1 \
  --cycle 168 \
  --n_clusters 32 \
  --top_k 4 \
  --itr 1

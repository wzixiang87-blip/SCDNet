#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

#!/bin/bash
model_name=SCDNet
root_path=${ROOT_PATH:-./dataset/traffic/}
data_path=${DATA_PATH:-traffic.csv}

echo "========== Traffic: pred_len = 96 =========="
python -u run.py \
  --is_training 1 \
  --root_path /root/autodl-tmp/all_datasets/traffic \
  --data_path traffic.csv \
  --model_id "traffic_96_96" \
  --model SCDNet \
  --data custom \
  --features M \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 862 \
  --des Exp \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 1 \
  --dropout 0.1 \
  --output_proj_dropout 0.0 \
  --batch_size 16 \
  --train_epochs 50 \
  --patience 5 \
  --learning_rate 0.001 \
  --lradj type3 \
  --cycle 168 \
  --n_clusters 64 \
  --top_k 6 \
  --itr 1

echo "========== Traffic: pred_len = 192 =========="
python -u run.py \
  --is_training 1 \
  --root_path /root/autodl-tmp/all_datasets/traffic \
  --data_path traffic.csv \
  --model_id "traffic_96_96" \
  --model SCDNet \
  --data custom \
  --features M \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 862 \
  --des Exp \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 1 \
  --dropout 0.1 \
  --output_proj_dropout 0.1 \
  --batch_size 16 \
  --train_epochs 50 \
  --patience 5 \
  --learning_rate 0.001 \
  --lradj type3 \
  --cycle 168 \
  --n_clusters 64 \
  --top_k 6 \
  --itr 1

echo "========== Traffic: pred_len = 336 =========="
python -u run.py \
  --is_training 1 \
  --root_path /root/autodl-tmp/all_datasets/traffic \
  --data_path traffic.csv \
  --model_id "traffic_96_96" \
  --model SCDNet \
  --data custom \
  --features M \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 862 \
  --des Exp \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 1 \
  --dropout 0.1 \
  --output_proj_dropout 0.0 \
  --batch_size 16 \
  --train_epochs 50 \
  --patience 5 \
  --learning_rate 0.001 \
  --lradj type3 \
  --cycle 168 \
  --n_clusters 64 \
  --top_k 6 \
  --itr 1

echo "========== Traffic: pred_len = 720 =========="
python -u run.py \
  --is_training 1 \
  --root_path /root/autodl-tmp/all_datasets/traffic \
  --data_path traffic.csv \
  --model_id "traffic_96_720" \
  --model SCDNet \
  --data custom \
  --features M \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 862 \
  --des Exp \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 2 \
  --dropout 0.1 \
  --output_proj_dropout 0.0 \
  --batch_size 8 \
  --train_epochs 50 \
  --patience 5 \
  --learning_rate 0.001 \
  --lradj type3 \
  --cycle 168 \
  --n_clusters 64 \
  --top_k 6 \
  --itr 1
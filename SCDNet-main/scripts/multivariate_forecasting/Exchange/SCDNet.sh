#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=SCDNet
root_path=${ROOT_PATH:-./dataset/exchange_rate/}
data_path=${DATA_PATH:-exchange_rate.csv}

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 根据你的实际环境修改这两个路径
root_path="./data/exchange_rate"
data_path="exchange_rate.csv"

model_name="SCDNet"   # 此处声明，若需要也可硬编码到命令中

echo "========== Exchange: pred_len = 96 =========="
python -u run.py \
  --is_training 1 \
  --root_path /root/autodl-tmp/all_datasets/exchange_rate \
  --data_path exchange_rate.csv \
  --model_id "Exchange_96_96" \
  --model SCDNet \
  --data custom \
  --features M \
  --freq d \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 8 \
  --des Exp \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 4 \
  --e_layers 2 \
  --dropout 0.1 \
  --output_proj_dropout 0.2 \
  --batch_size 32 \
  --train_epochs 20 \
  --patience 5 \
  --learning_rate 0.001 \
  --lradj type1 \
  --cycle 7 \
  --n_clusters 8 \
  --top_k 2 \
  --itr 1

echo "========== Exchange: pred_len = 192 =========="
python -u run.py \
  --is_training 1 \
  --root_path "${root_path}" \
  --data_path "${data_path}" \
  --model_id "Exchange_96_192" \
  --model "${model_name}" \
  --data custom \
  --features M \
  --freq d \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 8 \
  --des Exp \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 4 \
  --e_layers 2 \
  --dropout 0.1 \
  --output_proj_dropout 0.2 \
  --batch_size 32 \
  --train_epochs 20 \
  --patience 5 \
  --learning_rate 0.001 \
  --lradj type1 \
  --cycle 7 \
  --n_clusters 8 \
  --top_k 2 \
  --itr 1

echo "========== Exchange: pred_len = 336 =========="
python -u run.py \
  --is_training 1 \
  --root_path "${root_path}" \
  --data_path "${data_path}" \
  --model_id "Exchange_96_336" \
  --model "${model_name}" \
  --data custom \
  --features M \
  --freq d \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 8 \
  --des Exp \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 3 \
  --dropout 0.1 \
  --output_proj_dropout 0.2 \
  --batch_size 32 \
  --train_epochs 20 \
  --patience 5 \
  --learning_rate 0.0005 \
  --lradj type1 \
  --cycle 7 \
  --n_clusters 12 \
  --top_k 3 \
  --itr 1

echo "========== Exchange: pred_len = 720 =========="
python -u run.py \
  --is_training 1 \
  --root_path "${root_path}" \
  --data_path "${data_path}" \
  --model_id "Exchange_96_720" \
  --model "${model_name}" \
  --data custom \
  --features M \
  --freq d \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 8 \
  --des Exp \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 3 \
  --dropout 0.1 \
  --output_proj_dropout 0.2 \
  --batch_size 32 \
  --train_epochs 20 \
  --patience 5 \
  --learning_rate 0.0005 \
  --lradj type1 \
  --cycle 7 \
  --n_clusters 12 \
  --top_k 3 \
  --itr 1

export CUDA_VISIBLE_DEVICES=0

model_name=SCDNet

python -u run.py \
 --is_training 1
--root_path D:\projects\all_datasets\all_datasets\ETT-small
--data_path ETTm2.csv
--model_id ETTm2_96_96
--model SCDNet
--data ETTm2
--features M
--seq_len 96
--pred_len 96
--e_layers 1
--enc_in 7
--des 'Exp'
--d_model 512
--d_ff 2048
--n_heads 4
--output_proj_dropout 0.5
--itr 1
--cycle 96
--n_clusters 16
--top_k 4
--learning_rate 0.001







python -u run.py \
--is_training 1
--root_path D:\projects\all_datasets\all_datasets\ETT-small
--data_path ETTm2.csv
--model_id ETTm2_96_192
--model SCDNet
--data ETTm2
--features M
--seq_len 96
--pred_len 192
--e_layers 1
--enc_in 7
--des 'Exp'
--d_model 512
--d_ff 2048
--n_heads 4
--output_proj_dropout 0.5
--itr 1
--cycle 96
--n_clusters 16
--top_k 4
--learning_rate 0.001



python -u run.py \
  --is_training 1 \
  --root_path D:\projects\all_datasets\all_datasets\ETT-small
--data_path ETTm2.csv
--model_id ETTm2_96_336
--model SCDNet
--data ETTm2
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 1 \
  --enc_in 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 1024
  --n_heads 2\
  --output_proj_dropout 0.3 \
  --itr 1 \
  --cycle 96
--learning_rate 0.001





python -u run.py \
 --is_training 1
--root_path D:\projects\all_datasets\all_datasets\ETT-small
--data_path ETTm2.csv
--model_id ETTm2_96_720
--model SCDNet
--data ETTm2
--features M
--seq_len 96
--pred_len 720
--e_layers 1
--enc_in 7
--des 'Exp'
--d_model 512
--d_ff 2048
--n_heads 4
--output_proj_dropout 0.3
--itr 1
--cycle 96
--n_clusters 8
--top_k 3
--learning_rate 0.001
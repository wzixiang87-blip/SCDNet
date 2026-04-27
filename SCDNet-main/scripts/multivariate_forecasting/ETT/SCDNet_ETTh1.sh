export CUDA_VISIBLE_DEVICES=0

model_name=SCDNet

python -u run.py \
  --is_training 1
  --root_path D:\projects\all_datasets\all_datasets\ETT-small
  --data_path ETTh1.csv
  --model_id ETTh1_96_96
  --model SCDNet
  --data ETTh1
  --features M
  --seq_len 192
  --pred_len 96
  --e_layers 3
  --enc_in 7
  --des 'Exp'
  --d_model 512
  --n_heads 4
  --d_ff 2048
  --output_proj_dropout 0.3
  --itr 1
  --cycle 24
--n_clusters 16
--top_k 3
--learning_rate  0.001


python -u run.py \
--is_training 1
  --root_path D:\projects\all_datasets\all_datasets\ETT-small
  --data_path ETTh1.csv
  --model_id ETTh1_96_192
  --model SCDNet
  --data ETTh1
  --features M
  --seq_len 96
  --pred_len 192
--e_layers 3
--enc_in 7
--des 'Exp'
--d_model 512
--n_heads 4
--d_ff 2048
--output_proj_dropout 0.3
--itr 1
--cycle 24
--n_clusters 16
--top_k 4
--learning_rate 0.001


python -u run.py \
  --is_training 1
--root_path D:\projects\all_datasets\all_datasets\ETT-small
--data_path ETTh1.csv
--model_id ETTh1_96_336
--model SCDNet
--data ETTh1
--features M
--seq_len 96
--pred_len 336
--e_layers 3
--enc_in 7
--des 'Exp'
--d_model 512
--n_heads 4
--d_ff 2048
--output_proj_dropout 0.3
--itr 1
--cycle 24
--n_clusters 16
--top_k 4
--learning_rate 0.001

python -u run.py
 --is_training 1
--root_path D:\projects\all_datasets\all_datasets\ETT-small
--data_path ETTh1.csv
--model_id ETTh1_96_720
--model SCDNet
--data ETTh1
--features M
--seq_len 96
--pred_len 720
--e_layers 3
--enc_in 7
--des 'Exp'
--d_model 512
--n_heads 4
--d_ff 2048
--output_proj_dropout 0.2
--itr 1
--patience 3
--cycle 24
--n_clusters 16
--top_k 4
--dropout 0.1
--learning_rate 0.0005
w_time, w_freq, w_aux = 1.0, 0.4, 0.01

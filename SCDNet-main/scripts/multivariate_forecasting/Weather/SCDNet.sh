export CUDA_VISIBLE_DEVICES=0

model_name=SCDNet

python -u run.py \
  --is_training 1 \
--root_path /root/autodl-tmp/all_datasets/weather \
--data_path weather.csv \
--model_id weather_96_96 \
--model SCDNet
--data custom \
--features M \
--seq_len 96 \
--pred_len 96 \
--e_layers 2 \
--enc_in 21 \
--des 'Exp' \
--d_model 64 \
--d_ff 256 \
--output_proj_dropout 0.5 \
--n_heads 4 \
--learning_rate 0.001 \
--batch_size 32 \
--itr 1 \
--cycle 144 \
--n_clusters 16 \
--top_k 4 \
--patience 3


python -u run.py \
 --is_training
1
--root_path
D:\projects\all_datasets\all_datasets\weather
--data_path
weather.csv
--model_id
weather_96_192
--model
SCDNet
--data
custom
--features
M
--seq_len
96
--pred_len
192
--e_layers
1
--enc_in
21
--des
'Exp'
--d_model
256
--d_ff
1024
--output_proj_dropout
0.5
--n_heads
4
--learning_rate
0.001
--batch_size
64
--itr
1
--cycle
144
--n_clusters
16
--top_k
3
--patience
3

python -u run.py \
--is_training 1
--root_path D:\projects\all_datasets\all_datasets\weather
--data_path weather.csv
--model_id weather_96_336
--model SCDNetn
--data custom
--features M
--seq_len 96
--pred_len 336
--e_layers 1
--enc_in 21
--des 'Exp'
--d_model 128
--d_ff 128
--output_proj_dropout 0.5
--n_heads 4
--learning_rate 0.001
--batch_size 32
--itr 1
--cycle 144
--n_clusters 16
--top_k 3
--patience 3


python -u run.py \
--is_training 1 \
--root_path /root/autodl-tmp/all_datasets/weather \
--data_path weather.csv \
--model_id weather_96_720 \
--model SCDNet
--data custom \
--features M \
--seq_len 96 \
--pred_len 720 \
--e_layers 4 \
--enc_in 21 \
--des 'Exp' \
--d_model 256 \
--d_ff 1024 \
--output_proj_dropout 0.3 \
--n_heads 8 \
--learning_rate 0.001 \
--batch_size 128 \
--itr 1 \
--cycle 144 \
--dropout 0.2 \
--n_clusters 16 \
--top_k 4 \
--patience 3

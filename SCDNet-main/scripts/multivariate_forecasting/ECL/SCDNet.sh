export CUDA_VISIBLE_DEVICES=0

model_name=SCDNet

python -u run.py \
  --is_training 1 \
  --root_path all_datasets/electricity \
  --data_path electricity.csv \
  --model_id ECL_96_96_SCDNet_SCD \
  --model SCDNet \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --label_len 48 \
  --enc_in 321 \
  --n_heads 4 \
  --e_layers 3 \
  --d_model 512 \
  --d_ff 1024 \
  --dropout 0.1 \
  --output_proj_dropout 0.1 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --train_epochs 10\
  --patience 3 \
  --itr 1 \
  --n_clusters 64 \
  --top_k 16 \
  --d_scd 64 \
    --cycle 168




export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --root_path  all_datasets/electricity \
  --data_path electricity.csv \
  --model SCDNet \
  --model_id ECL_96_192_ema_scd_v2 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 321 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --dropout 0.1 \
  --output_proj_dropout 0.1 \
  --batch_size 16 \
  --learning_rate 0.005 \
  --train_epochs 10 \
  --cycle 168 \
  --n_clusters 150 \
  --top_k 4 \
  --d_scd 64 \
  --itr 1




export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
   --root_path  all_datasets/electricity \
  --data_path electricity.csv \
  --model SCDNet \
  --model_id ECL_96_336_SCDNet \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 321 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 4 \
  --dropout 0.1 \
  --output_proj_dropout 0.1 \
  --batch_size 16 \
  --learning_rate 0.005 \
  --train_epochs 10 \
  --cycle 168 \
  --n_clusters 32 \
  --top_k 4 \
  --d_scd 64 \
  --itr 1






export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --root_path  all_datasets/electricity \
  --data_path electricity.csv \
  --model SCDNet \
  --model_id ECL_96_720_SCDNet \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 321 \
   --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --dropout 0.2 \
  --output_proj_dropout 0.2 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --cycle 168 \
  --n_clusters 32 \
  --top_k 3 \
  --d_scd 64 \
  --itr 1

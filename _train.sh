#!/bin/bash

#### local path
SQUAD_DIR=~/SQUAD
INIT_CKPT_DIR=~/checkpoint/xlnet_cased_L-24_H-1024_A-16
PROC_DATA_DIR=~/drive/proc_data/squad
MODEL_DIR=~/checkpoint/squad-xlnet/squad

python run_squad_3.py \
  --do_train=True \
  --do_predict=True \
  --use_tpu=False \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --model_config_path=${INIT_CKPT_DIR}/xlnet_config.json \
  --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
  --output_dir=${PROC_DATA_DIR} \
  --init_checkpoint=${INIT_CKPT_DIR}/xlnet_model.ckpt \
  --model_dir=${MODEL_DIR} \
  --train_file=${SQUAD_DIR}/train-v2.0.json \
  --predict_file=${SQUAD_DIR}/dev-v2.0.json \
  --uncased=False \
  --max_seq_length=340 \
  --train_batch_size=4 \
  --predict_batch_size=32 \
  --learning_rate=2e-5 \
  --adam_epsilon=1e-6 \
  --iterations=1000 \
  --save_steps=5000 \
  --train_steps=90000 \
  --warmup_steps=1000 \
  $@

# ~/desligar.sh
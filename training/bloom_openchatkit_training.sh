# Copyright 2023 SambaNova Systems, Inc.

# Licensed under the BLOOMChat License, Version 1.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

# https://github.com/sambanova/bloomchat/blob/main/LICENSE

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions, limitations and restrictions under the License.

#!/bin/bash
PEF=$1
SCRIPT=transformers_hook.py
OUTPUT_DIR=./ckpt/bloom_ock_100k/lr_1e-5/
NUM_TRAIN_EPOCHS=1
WARMUP_STEPS=0
LOG_STEPS=1
SAVE_STEPS=100
SAVE_INTERVAL=1
EVAL_STEPS=500000
SKIP_STEPS=0
DATA_DIR=../dataset_prep/bloom_ock_100k_out/hdf5
VENV_PATH=$2
CACHE_DIR=./cache
MODEL_PATH=bigscience/bloom
export OMP_NUM_THREADS=8
export SAMBA_CCL_ASYNC_ALLREDUCE=1
export SAMBA_CCL_HIERARCHICAL_ALLREDUCE=0
export SF_RNT_LOG_LEVEL=ERR
source ${VENV_PATH}/bin/activate && /opt/mpich-3.4.3/bin/mpirun -np 8 --host host1,host2,host3,host4,host5,host6,host7,host8 python3 -u $SCRIPT run --module_name gpt2_pretrain --task_name clm --max_seq_length 2048 -b 16 --output_dir=$OUTPUT_DIR --overwrite_output_dir --per_device_train_batch_size 16 --do_train --data_dir ${DATA_DIR} --num_train_epochs $NUM_TRAIN_EPOCHS --non_split_head --model_name_or_path $MODEL_PATH --tokenizer_name bigscience/bloom --skip_broadcast_patch --no_index_select_patch  --reduce-on-rdu  --max_grad_norm_clip 1.0 --weight_decay 0.1 --learning_rate 1e-5 --pef $PEF --logging_steps $LOG_STEPS --save_steps ${SAVE_STEPS} --skip_steps $SKIP_STEPS --use_diagonal_mask --use_token_type_ids --prompt_loss_weight 0.0 --cache_dir ${CACHE_DIR} --model_name bloom --data-parallel --save_from_rdu --warmup_steps ${WARMUP_STEPS} --max_steps -1 --save_strategy steps --save_weights_only

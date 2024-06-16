#!/bin/sh

export GPUS_PER_NODE=$(nvidia-smi --list-gpus|wc -l)
export LOCAL_WORLD_SIZE=$GPUS_PER_NODE

# WORD_SIZE is incorrectly set as number of nodes by Flyte pytorch plugin
export NUM_NODES=$WORLD_SIZE

# This sets correct for the world size
export WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

# Replace model and data path with your own path
torchrun --nnodes=$NUM_NODES --nproc-per-node=$GPUS_PER_NODE \
      training.py \
      --model_path /shared/public/models/Meta-Llama-3-8B \
      --data_path /shared/public/data/mmlu \
      --max_steps 200 \
      --bf16 true \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 8 \
      --max_seq_length 2048 \
      --save_strategy "no" \
      --learning_rate 1e-6 \
      --logging_steps 1 \
      --include_num_input_tokens_seen \
      --report_to none \
      --output_dir /tmp/output \
      --fsdp "full_shard auto_wrap" \
      --fsdp_config config/fsdp_config.json \
      --torch_compile true

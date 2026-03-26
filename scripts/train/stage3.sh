#!/bin/bash
# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0

# ===== Load environment config =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_CONFIG="$PROJECT_ROOT/env_config.yaml"

parse_yaml() {
    local yaml_file=$1
    local key=$2
    grep "^${key}:" "$yaml_file" | sed "s/^${key}: *\"\(.*\)\"/\1/" | sed "s/^${key}: *'\(.*\)'/\1/" | sed "s/^${key}: *//"
}

if [ ! -f "$ENV_CONFIG" ]; then
    echo "Error: env_config.yaml not found at $ENV_CONFIG"
    echo "Copy env_config_example.yaml to env_config.yaml and fill in your paths."
    exit 1
fi

# Load paths from config
VLM_PATH=$(parse_yaml "$ENV_CONFIG" "vlm_path")
VGEN_MODEL_PATH=$(parse_yaml "$ENV_CONFIG" "vgen_model_path")
VAE_PATH=$(parse_yaml "$ENV_CONFIG" "vae_path")
PRE_T5_CONTEXT_PATH=$(parse_yaml "$ENV_CONFIG" "pre_t5_context_path")
OUTPUT_BASE_DIR=$(parse_yaml "$ENV_CONFIG" "train_output_dir")

cd $PROJECT_ROOT

JOB_NAME="lightfusion_stage3"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${JOB_NAME}"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR

WANDB_PROJECT="lightfusion"
WANDB_NAME=$JOB_NAME
WANDB_RUN_ID="1"

# replace the variables with your own.
num_nodes=4
node_rank=0
nproc_per_node=8
master_addr=127.0.0.1
master_port=12345

# TODO: update total_steps
torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=$nproc_per_node \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file data/configs/stage3.yaml \
  --vlm_path $VLM_PATH \
  --vgen_model_path $VGEN_MODEL_PATH \
  --vae_path $VAE_PATH \
  --pre_t5_context_path $PRE_T5_CONTEXT_PATH \
  --vgen_task "ti2v-5B" \
  --layer_module Qwen2_5_VLDecoderLayer \
  --resume_from ${OUTPUT_BASE_DIR}/lightfusion_stage2/checkpoints/0015000 \
  --resume_model_only \
  --global_seed 2025 \
  --expected_num_tokens 16384 \
  --max_num_tokens_per_sample 8192 \
  --max_num_tokens 20480 \
  --max_buffer_size 4 \
  --visual_und True \
  --visual_gen True \
  --mm_attn_qk_norm True \
  --llm_qk_norm False \
  --tie_word_embeddings False \
  --standalone_num_vlm_layers 0 \
  --use_vgen_for_mm_attn True \
  --freeze_und True \
  --zero_init_mm_attn True \
  --total_steps 15000 \
  --text_cond_dropout_prob 0.1 \
  --vae_cond_dropout_prob 0.1 \
  --vit_cond_dropout_prob 0.5 \
  --timestep_shift 4 \
  --log_every 1 \
  --warmup_steps 2000 \
  --lr_scheduler constant \
  --lr 3e-5 \
  --beta1 0.9 \
  --beta2 0.95 \
  --eps 1e-15 \
  --num_replicate $num_nodes \
  --num_shard $nproc_per_node \
  --num_workers 1 \
  --sharding_strategy HYBRID_SHARD \
  --use_orig_params True \
  --results_dir ${OUTPUT_DIR} \
  --checkpoint_dir ${CHECKPOINT_DIR} \
  --wandb_project $WANDB_PROJECT \
  --wandb_name $WANDB_NAME \
  --wandb_runid $WANDB_RUN_ID \
  --wandb_offline False \
  2>&1 | tee "$OUTPUT_DIR/training_log.txt"

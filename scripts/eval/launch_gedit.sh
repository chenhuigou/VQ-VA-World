#!/bin/bash
# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0
#
# GEdit-Bench evaluation script (Base, no think)
# Usage: bash scripts/eval/launch_gedit.sh [MODEL_NAME] [CKPT]

set -x

# ===== Load environment config =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_CONFIG="$PROJECT_ROOT/env_config.yaml"

parse_yaml() {
    python3 -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print(c.get(sys.argv[2],''))" "$1" "$2"
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
CHECKPOINT_BASE_DIR=$(parse_yaml "$ENV_CONFIG" "checkpoint_base_dir")
GEDIT_DATA_PATH=$(parse_yaml "$ENV_CONFIG" "gedit_data_path")
OUTPUT_BASE_DIR=$(parse_yaml "$ENV_CONFIG" "output_base_dir")

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ===== Configuration =====
GPUS=${ARNOLD_WORKER_GPU:-8}
BENCHMARK="gedit_bench"
MODEL_NAME="${1:-lightfusion}"
CKPT="${2:-0070000}"
POSTFIX="ema_magic_ts4_ss3_512"

LOAD_FROM="$CHECKPOINT_BASE_DIR/$MODEL_NAME/checkpoints/$CKPT"

# Local tmp for output, then copy to HDFS
LOCAL_BASE_DIR="/tmp/gedit_bench_eval"
LOCAL_OUTPUT_DIR="$LOCAL_BASE_DIR/$MODEL_NAME/step${CKPT}_${POSTFIX}"
LOCAL_GEN_DIR="$LOCAL_OUTPUT_DIR/gen_image"
LOCAL_LOG_DIR="$LOCAL_OUTPUT_DIR/logs"
HDFS_OUTPUT_DIR="$OUTPUT_BASE_DIR/$BENCHMARK/$MODEL_NAME/step${CKPT}_${POSTFIX}"

mkdir -p "$LOCAL_OUTPUT_DIR" "$LOCAL_GEN_DIR" "$LOCAL_LOG_DIR"

# ===== Image Generation =====
for ((i=0; i<$GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=${i} nohup python \
    ./eval/i2i/i2i_inference.py \
    --benchmark $BENCHMARK \
    --load_from "$LOAD_FROM" \
    --data_path "$GEDIT_DATA_PATH" \
    --output_dir "$LOCAL_GEN_DIR" \
    --vlm_path "$VLM_PATH" \
    --vgen_model_path "$VGEN_MODEL_PATH" \
    --vae_path "$VAE_PATH" \
    --pre_t5_context_path "$PRE_T5_CONTEXT_PATH" \
    --mm_attn_qk_norm \
    --vae_transform_sizes 1024 512 32 \
    --vit_transform_sizes 532 224 28 \
    --task ti2v-5B \
    --num_timesteps 50 \
    --timestep_shift 4 \
    --cfg_text_scale 3 \
    --cfg_img_scale 1 \
    --use_ema \
    --use_magic_negative_prompt \
    --use_vgen_for_mm_attn \
    --rank $i \
    --world_size $GPUS \
    2>&1 | tee "$LOCAL_LOG_DIR/${i}.log" &
done
wait
echo "Image Generation Done for ${MODEL_NAME}_${CKPT}"

# ===== Evaluation =====
python ./eval/i2i/GEdit-Bench/test_gedit_score.py \
    --model_name "${MODEL_NAME}_${CKPT}" --save_path "$LOCAL_OUTPUT_DIR" \
    2>&1 | tee "$LOCAL_LOG_DIR/score.log"

python ./eval/i2i/GEdit-Bench/calculate_statistics.py \
    --model_name "${MODEL_NAME}_${CKPT}" --save_path "$LOCAL_OUTPUT_DIR" --language cn \
    2>&1 | tee "$LOCAL_LOG_DIR/cn_stats.log"

python ./eval/i2i/GEdit-Bench/calculate_statistics.py \
    --model_name "${MODEL_NAME}_${CKPT}" --save_path "$LOCAL_OUTPUT_DIR" --language en \
    2>&1 | tee "$LOCAL_LOG_DIR/en_stats.log"

# ===== Copy results to HDFS =====
mkdir -p "$HDFS_OUTPUT_DIR"
cp -r "$LOCAL_OUTPUT_DIR"/* "$HDFS_OUTPUT_DIR"/

echo "Score Compute Done for ${MODEL_NAME}_${CKPT}"

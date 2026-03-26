#!/bin/bash
# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0
#
# IMGEdit evaluation script (Base, no think)
# Usage: bash scripts/eval/launch_imgedit.sh [MODEL_NAME] [CKPT]

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
IMGEDIT_DATA_PATH=$(parse_yaml "$ENV_CONFIG" "imgedit_data_path")
OUTPUT_BASE_DIR=$(parse_yaml "$ENV_CONFIG" "output_base_dir")
OPENAI_API_KEY_VAL=$(parse_yaml "$ENV_CONFIG" "openai_api_key")
OPENAI_BASE_URL_VAL=$(parse_yaml "$ENV_CONFIG" "openai_base_url")
HTTP_PROXY_VAL=$(parse_yaml "$ENV_CONFIG" "http_proxy")
HTTPS_PROXY_VAL=$(parse_yaml "$ENV_CONFIG" "https_proxy")

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ===== Load API keys from api_keys.yaml =====
API_KEYS_FILE="$PROJECT_ROOT/api_keys.yaml"
if [ -f "$API_KEYS_FILE" ]; then
    DEFAULT_KEY=$(grep "^default_key:" "$API_KEYS_FILE" | sed 's/default_key: *"\(.*\)"/\1/')
    OPENAI_BASE_URL_KEYS=$(grep "^openai_base_url:" "$API_KEYS_FILE" | sed 's/openai_base_url: *"\(.*\)"/\1/')
    echo "Loaded default_key from api_keys.yaml"
    [ -n "$OPENAI_BASE_URL_KEYS" ] && OPENAI_BASE_URL_VAL="$OPENAI_BASE_URL_KEYS"
else
    echo "Warning: api_keys.yaml not found, using env_config default key"
    DEFAULT_KEY="$OPENAI_API_KEY_VAL"
fi

EVAL_API_KEY="${DEFAULT_KEY:-$OPENAI_API_KEY_VAL}"

# ===== Proxy & API setup =====
[ -n "$HTTP_PROXY_VAL" ] && export http_proxy="$HTTP_PROXY_VAL"
[ -n "$HTTPS_PROXY_VAL" ] && export https_proxy="$HTTPS_PROXY_VAL"
[ -n "$EVAL_API_KEY" ] && export OPENAI_API_KEY="$EVAL_API_KEY"

# ===== Configuration =====
GPUS=${ARNOLD_WORKER_GPU:-8}
BENCHMARK="imgedit_bench"
MODEL_NAME="${1:-lightfusion}"
CKPT="${2:-0070000}"
POSTFIX="ema_magic_ts4_ss3_512"

LOAD_FROM="$CHECKPOINT_BASE_DIR/$MODEL_NAME/checkpoints/$CKPT"

# Local tmp for output, then copy to HDFS
LOCAL_BASE_DIR="/tmp/imgedit_bench_eval"
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
    --data_path "$IMGEDIT_DATA_PATH" \
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
ORIGIN_IMG_ROOT="${IMGEDIT_DATA_PATH}/data/Benchmark/singleturn"

python3 ./eval/i2i/imgedit/basic_bench.py \
    --result_img_folder "${LOCAL_GEN_DIR}" \
    --result_json "${LOCAL_OUTPUT_DIR}/imgedit_bench.json" \
    --edit_json "${IMGEDIT_DATA_PATH}/eval_prompts/basic_edit.json" \
    --prompts_json "${IMGEDIT_DATA_PATH}/eval_prompts/prompts.json" \
    --origin_img_root "${ORIGIN_IMG_ROOT}" \
    --num_processes 32 \
    --api_key "${EVAL_API_KEY}" \
    ${OPENAI_BASE_URL_VAL:+--base_url "${OPENAI_BASE_URL_VAL}"}

python3 ./eval/i2i/imgedit/step1_get_avgscore.py \
    --result_json "${LOCAL_OUTPUT_DIR}/imgedit_bench.json" \
    --average_score_json "${LOCAL_OUTPUT_DIR}/score.json"

# ===== Copy results to HDFS =====
mkdir -p "$HDFS_OUTPUT_DIR"
cp -r "$LOCAL_OUTPUT_DIR"/* "$HDFS_OUTPUT_DIR"/

echo "Evaluation Done for ${MODEL_NAME}_${CKPT}"

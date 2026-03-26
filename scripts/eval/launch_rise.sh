#!/bin/bash
# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0
#
# RISE-Bench evaluation script (Think Gen)
# Usage: bash scripts/eval/launch_rise.sh [MODEL_NAME] [CKPT]

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
    exit 1
fi

# Load paths from config
VLM_PATH=$(parse_yaml "$ENV_CONFIG" "vlm_path")
VGEN_MODEL_PATH=$(parse_yaml "$ENV_CONFIG" "vgen_model_path")
VAE_PATH=$(parse_yaml "$ENV_CONFIG" "vae_path")
PRE_T5_CONTEXT_PATH=$(parse_yaml "$ENV_CONFIG" "pre_t5_context_path")
CHECKPOINT_BASE_DIR=$(parse_yaml "$ENV_CONFIG" "checkpoint_base_dir")
RISE_DATA_PATH=$(parse_yaml "$ENV_CONFIG" "rise_data_path")
OUTPUT_BASE_DIR=$(parse_yaml "$ENV_CONFIG" "output_base_dir")
OPENAI_API_KEY_VAL=$(parse_yaml "$ENV_CONFIG" "openai_api_key")
HTTP_PROXY_VAL=$(parse_yaml "$ENV_CONFIG" "http_proxy")
HTTPS_PROXY_VAL=$(parse_yaml "$ENV_CONFIG" "https_proxy")
RISE_EVAL_BASE=$(parse_yaml "$ENV_CONFIG" "rise_eval_base")
RISE_JSON_PATH=$(parse_yaml "$ENV_CONFIG" "rise_json_path")

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ===== Load API keys =====
API_KEYS_FILE="$PROJECT_ROOT/api_keys.yaml"
if [ -f "$API_KEYS_FILE" ]; then
    mapfile -t API_KEYS < <(grep '^ *- "' "$API_KEYS_FILE" | sed 's/^ *- "\(.*\)"/\1/')
    OPENAI_BASE_URL_KEYS=$(grep "^openai_base_url:" "$API_KEYS_FILE" | sed 's/openai_base_url: *"\(.*\)"/\1/')
    echo "Loaded ${#API_KEYS[@]} API keys from api_keys.yaml"
else
    API_KEYS=("$OPENAI_API_KEY_VAL")
    OPENAI_BASE_URL_KEYS=""
fi

# ===== Proxy & API setup =====
[ -n "$HTTP_PROXY_VAL" ] && export http_proxy="$HTTP_PROXY_VAL"
[ -n "$HTTPS_PROXY_VAL" ] && export https_proxy="$HTTPS_PROXY_VAL"
[ ${#API_KEYS[@]} -gt 0 ] && export OPENAI_API_KEY="${API_KEYS[0]}"
[ -n "$OPENAI_BASE_URL_KEYS" ] && export OPENAI_BASE_URL="$OPENAI_BASE_URL_KEYS"

# ===== Configuration =====
GPUS=${ARNOLD_WORKER_GPU:-8}
BENCHMARK="rise_bench"
MODEL_NAME="${1:-lightfusion}"
CKPT="${2:-0070000}"
POSTFIX="think_4_4_2_renorm_1024_512"

LOAD_FROM="$CHECKPOINT_BASE_DIR/$MODEL_NAME/checkpoints/$CKPT"

# Use local temp dir, then copy to output
LOCAL_BASE_DIR="/tmp/rise_bench_eval"
LOCAL_OUTPUT_DIR="$LOCAL_BASE_DIR/$MODEL_NAME/step${CKPT}_$POSTFIX"
LOCAL_GEN_DIR="$LOCAL_OUTPUT_DIR/gen_image"
LOCAL_LOG_DIR="$LOCAL_OUTPUT_DIR/logs"
HDFS_OUTPUT_DIR="$OUTPUT_BASE_DIR/$BENCHMARK/$MODEL_NAME/step${CKPT}_$POSTFIX"

mkdir -p "$LOCAL_OUTPUT_DIR" "$LOCAL_GEN_DIR" "$LOCAL_LOG_DIR"

echo "[$(date)] Starting image generation for ${MODEL_NAME}_${CKPT}"

# ===== Image Generation (Think mode) =====
for ((j=0; j<$GPUS; j++)); do
    CUDA_VISIBLE_DEVICES=${j} nohup python \
    ./eval/i2i/i2i_inference_think.py \
    --benchmark $BENCHMARK \
    --load_from "$LOAD_FROM" \
    --data_path "$RISE_DATA_PATH" \
    --output_dir "$LOCAL_GEN_DIR" \
    --vlm_path "$VLM_PATH" \
    --vgen_model_path "$VGEN_MODEL_PATH" \
    --vae_path "$VAE_PATH" \
    --pre_t5_context_path "$PRE_T5_CONTEXT_PATH" \
    --mm_attn_qk_norm \
    --mm_attn_num_layer_type max \
    --vae_transform_sizes 1024 512 32 \
    --vit_transform_sizes 532 224 28 \
    --standalone_num_vlm_layers 0 \
    --task ti2v-5B \
    --num_timesteps 50 \
    --timestep_shift 4 \
    --cfg_text_scale 4 \
    --cfg_img_scale 2 \
    --cfg_renorm_type text_channel \
    --cfg_interval 0.7 1.0 \
    --use_ema \
    --use_magic_negative_prompt \
    --use_vgen_for_mm_attn \
    --think \
    --seed 43 \
    --max_length 2048 \
    --rank $j \
    --world_size $GPUS \
    2>&1 | tee "$LOCAL_LOG_DIR/${j}.log" &
done
wait
echo "[$(date)] Image Generation Done for ${MODEL_NAME}_${CKPT}"

# ===== GPT Evaluation =====
if [ -n "$RISE_EVAL_BASE" ] && [ -f "$RISE_EVAL_BASE/gen/rise/gpt_eval.py" ]; then
    echo "[$(date)] Running RISE GPT evaluation..."
    cd "$RISE_EVAL_BASE"

    DATA="$RISE_JSON_PATH"
    IN="$RISE_DATA_PATH"
    OUT="$LOCAL_GEN_DIR"
    NUM_SHARDS=6

    for i in $(seq 0 $((NUM_SHARDS-1))); do
        KEY_IDX=$((i % ${#API_KEYS[@]}))
        SELECTED_KEY="${API_KEYS[$KEY_IDX]}"

        OPENAI_API_KEY="$SELECTED_KEY" \
        timeout 7200 python ./gen/rise/gpt_eval.py \
            --data "$DATA" \
            --input "$IN" \
            --output "$OUT" \
            --mode shard \
            --num_shards $NUM_SHARDS \
            --shard_id $i \
            --nproc 6 \
            > "$LOCAL_LOG_DIR/gpt_eval_shard_${i}.log" 2>&1 &
    done

    wait
    echo "[$(date)] All shards completed"

    python ./gen/rise/gpt_eval.py \
        --data "$DATA" \
        --input "$IN" \
        --output "$OUT" \
        --mode finalize

    cd "$PROJECT_ROOT"
    echo "[$(date)] RISE Evaluation Done for ${MODEL_NAME}_${CKPT}"
else
    echo "[$(date)] Skipping RISE GPT evaluation: gpt_eval.py not found."
fi

echo "[$(date)] Copying results to output dir..."
mkdir -p "$HDFS_OUTPUT_DIR"
cp -r "$LOCAL_OUTPUT_DIR"/* "$HDFS_OUTPUT_DIR"/

echo "[$(date)] All tasks completed for ${MODEL_NAME}_${CKPT}"
echo "Results saved to: $HDFS_OUTPUT_DIR"

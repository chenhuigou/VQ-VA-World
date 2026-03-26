#!/bin/bash
# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0
#
# GenEval evaluation script
# Usage: bash scripts/eval/launch_geneval.sh [MODEL_NAME] [CKPT]

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
OUTPUT_BASE_DIR=$(parse_yaml "$ENV_CONFIG" "output_base_dir")

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ===== Install dependencies =====
pip install open-clip-torch
pip install clip-benchmark
pip install --upgrade setuptools

pip install -U openmim
mim install mmengine mmcv-full==1.7.2

cd eval/t2i/geneval
if [ ! -d mmdetection ]; then
    git clone https://github.com/open-mmlab/mmdetection.git
fi
cd mmdetection; git checkout 2.x
pip install -v -e .
cd "$PROJECT_ROOT"

bash ./eval/t2i/geneval/evaluation/download_models.sh ./eval/t2i/geneval/model

# ===== Configuration =====
GPUS=${ARNOLD_WORKER_GPU:-8}
BENCHMARK="geneval"
MODEL_NAME="${1:-lightfusion}"
CKPT="${2:-0070000}"
POSTFIX="ema_magic"

LOAD_FROM="$CHECKPOINT_BASE_DIR/$MODEL_NAME/checkpoints/$CKPT"

# Local tmp for output, then copy to HDFS
LOCAL_BASE_DIR="/tmp/geneval_eval"
LOCAL_OUTPUT_DIR="$LOCAL_BASE_DIR/$MODEL_NAME/step${CKPT}_${POSTFIX}"
LOCAL_LOG_DIR="$LOCAL_OUTPUT_DIR/logs"
HDFS_OUTPUT_DIR="$OUTPUT_BASE_DIR/$BENCHMARK/$MODEL_NAME/step${CKPT}_${POSTFIX}"

mkdir -p "$LOCAL_OUTPUT_DIR" "$LOCAL_LOG_DIR"

# ===== Image Generation =====
for ((i=0; i<$GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=${i} nohup python3 \
    ./eval/t2i/t2i_inference.py \
    --benchmark $BENCHMARK \
    --load_from "$LOAD_FROM" \
    --output_dir "$LOCAL_OUTPUT_DIR" \
    --metadata_file ./eval/t2i/geneval/prompts/evaluation_metadata_long.jsonl \
    --num_images 4 \
    --batch_size 1 \
    --vlm_path "$VLM_PATH" \
    --vgen_model_path "$VGEN_MODEL_PATH" \
    --vae_path "$VAE_PATH" \
    --pre_t5_context_path "$PRE_T5_CONTEXT_PATH" \
    --mm_attn_qk_norm \
    --sizes 1 1024 1024 \
    --task ti2v-5B \
    --num_timesteps 50 \
    --timestep_shift 4 \
    --cfg_scale 3 \
    --use_ema \
    --use_magic_negative_prompt \
    --use_vgen_for_mm_attn \
    --rank $i \
    --world_size $GPUS \
    2>&1 | tee "$LOCAL_LOG_DIR/${i}.log" &
done
wait
echo "Image Generation Done for ${MODEL_NAME}_${CKPT}"

# ===== Calculate score =====
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/t2i/geneval/evaluation/evaluate_images_mp.py \
    "$LOCAL_OUTPUT_DIR" \
    --outfile "$LOCAL_OUTPUT_DIR/results.jsonl" \
    --model-path ./eval/t2i/geneval/model

python ./eval/t2i/geneval/evaluation/summary_scores.py "$LOCAL_OUTPUT_DIR/results.jsonl"

# ===== Copy results to HDFS =====
mkdir -p "$HDFS_OUTPUT_DIR"
cp -r "$LOCAL_OUTPUT_DIR"/* "$HDFS_OUTPUT_DIR"/

echo "GenEval Evaluation Done for ${MODEL_NAME}_${CKPT}"

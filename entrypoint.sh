#!/bin/bash

set -e

# The container is dedicated to the pre-converted IndexTTS-2 vLLM model.
MODEL_DIR=${MODEL_DIR:-"/app/checkpoints/IndexTTS-2-vLLM"}
MODEL=${MODEL:-"kusuriuri/IndexTTS-2-vLLM"}
DOWNLOAD_MODEL=${DOWNLOAD_MODEL:-1}
PORT=${PORT:-9009}

required_model_files=(
    "config.yaml"
    "bpe.model"
    "gpt.pth"
    "s2mel.pth"
    "wav2vec2bert_stats.pt"
    "gpt/config.json"
    "bigvgan/config.json"
    "bigvgan/bigvgan_generator.pt"
    "campplus/campplus_cn_common.bin"
    "semantic_codec/model.safetensors"
)

missing_model_files() {
    local file
    for file in "${required_model_files[@]}"; do
        [ -f "$MODEL_DIR/$file" ] || echo "$file"
    done
}

model_exists() {
    [ -d "$MODEL_DIR" ] && [ -z "$(missing_model_files)" ]
}

download_model() {
    mkdir -p "$MODEL_DIR"
    echo "Downloading IndexTTS-2 model from ModelScope: $MODEL"
    modelscope download --model "$MODEL" --local_dir "$MODEL_DIR"
}

echo "Starting IndexTTS-2 API server..."
echo "Model directory: $MODEL_DIR"

if ! model_exists; then
    if [ "$DOWNLOAD_MODEL" != "1" ]; then
        echo "Missing IndexTTS-2 files in $MODEL_DIR:" >&2
        missing_model_files >&2
        exit 1
    fi
    download_model
fi

if ! model_exists; then
    echo "Model download completed but required files are still missing:" >&2
    missing_model_files >&2
    exit 1
fi

exec python3 api_server_v2.py \
    --model_dir "$MODEL_DIR" \
    --port "$PORT" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.25}" \
    --qwenemo_gpu_memory_utilization "${QWENEMO_GPU_MEMORY_UTILIZATION:-0.10}"

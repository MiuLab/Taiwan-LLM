#!/bin/sh

# 透過參數設定或預設值來定義變數
model=${1:-"yentinglin/Llama-3-Taiwan-70B-Instruct"}
port=${2:-8000}
gpus=${3:-'"device=0,1"'}
hf_token=${HF_TOKEN:-""}

# 使用 Docker 執行指令，進行模型的運算
sudo docker run --gpus $gpus \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$hf_token" \
    -p $port:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model $model \
    -tp 2 \

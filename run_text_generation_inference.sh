#!/bin/sh

# 透過參數設定或預設值來定義變數
model=${1:-"yentinglin/Taiwan-LLaMa-v1.0"}
num_shard=${2:-8}
volume=${3:-"$PWD/data"}
port=${4:-8080}
max_input_length=${5:-2000}
max_length=${6:-4000}

# 使用 Docker 執行指令，進行模型的運算
docker run --gpus all \
           --shm-size 1g \
           -p $port:80 \
           -v $volume:/data \
           ghcr.io/huggingface/text-generation-inference:latest \
           --model-id $model \
           --num-shard $num_shard \
           --max-input-length $max_input_length \
           --max-total-tokens $max_length

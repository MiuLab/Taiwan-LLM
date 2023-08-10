#!/bin/sh

model=${1:-"yentinglin/Taiwan-LLaMa"}
num_shard=${2:-8}  # Use the second command-line argument if provided, otherwise default to 8
volume=${3:-"$PWD/data"}  # Use the third command-line argument if provided, otherwise default to $PWD/data
port=${4:-8080}  # Use the fourth command-line argument if provided, otherwise default to 8080
max_input_length=${5:-2000}  # set to max prompt length (should be < max_length)
max_length=${6:-4000}  # set to max length in tokenizer_config

docker run --gpus all --shm-size 1g -p $port:80 $model_volume -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id $model --num-shard $num_shard --max-input-length $max_input_length --max-total-tokens $max_length

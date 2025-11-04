#!/bin/bash

# RunPod startup script for Llama 70B model
echo "Starting Llama 70B vLLM server on RunPod..."

# Set environment variables
export HF_HOME=/runpod-volume/cache
export TRANSFORMERS_CACHE=/runpod-volume/cache
export HF_DATASETS_CACHE=/runpod-volume/cache

# Create cache directory on persistent volume
mkdir -p /runpod-volume/cache

# Start vLLM server with Llama 70B (requires tensor parallelism)
python -m vllm.entrypoints.openai.api_server \
    --model=meta-llama/Llama-3.1-70B-Instruct \
    --host=0.0.0.0 \
    --port=8000 \
    --trust-remote-code \
    --download-dir=/runpod-volume/cache \
    --max-model-len=4096 \
    --tensor-parallel-size=2 \
    --gpu-memory-utilization=0.95 \
    --swap-space=8
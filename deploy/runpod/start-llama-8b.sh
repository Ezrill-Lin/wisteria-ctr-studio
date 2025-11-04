#!/bin/bash

# RunPod startup script for Llama 8B model
echo "Starting Llama 8B vLLM server on RunPod..."

# Set environment variables
export HF_HOME=/runpod-volume/cache
export TRANSFORMERS_CACHE=/runpod-volume/cache
export HF_DATASETS_CACHE=/runpod-volume/cache

# Create cache directory on persistent volume
mkdir -p /runpod-volume/cache

# Start vLLM server with Llama 8B
python -m vllm.entrypoints.openai.api_server \
    --model=meta-llama/Llama-3.1-8B-Instruct \
    --host=0.0.0.0 \
    --port=8000 \
    --trust-remote-code \
    --download-dir=/runpod-volume/cache \
    --max-model-len=4096 \
    --gpu-memory-utilization=0.90 \
    --swap-space=4
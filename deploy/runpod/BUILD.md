# Building Pre-baked vLLM Docker Images for RunPod

This directory contains a Dockerfile that pre-downloads the model weights during build time, so your RunPod serverless endpoint starts instantly without downloading from Hugging Face.

## Prerequisites

1. **Docker** installed and running
2. **Hugging Face token** with access to gated models (if using Llama)
3. **Docker Hub account** (or other container registry)

## Quick Start

### 1. Build the Image (8B Model)

```powershell
# Build for Llama 3.1 8B
docker build
  --build-arg MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
  --build-arg HF_TOKEN=$env:HF_TOKEN `
  -t ezrill/vllm-llama-8b:latest
  -f Dockerfile
```

### 2. Build the Image (70B Model)

```powershell
# Build for Llama 3.1 70B
docker build
  --build-arg MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
  --build-arg HF_TOKEN=$env:HF_TOKEN
  -t ezrill/vllm-llama-70b:latest
  -f Dockerfile
```

### 3. Push to Docker Hub

```powershell
# Login to Docker Hub
docker login

# Push the image
docker push your-dockerhub-username/vllm-llama-8b:latest
docker push your-dockerhub-username/vllm-llama-70b:latest
```

### 4. Use in RunPod

When creating a serverless endpoint in RunPod:

1. Go to **Serverless** → **New Endpoint**
2. Under **Container Image**, enter: `your-dockerhub-username/vllm-llama-8b:latest`
3. Set **Container Disk**: 20GB minimum (40GB for 70B)
4. Select GPU: RTX 4090 for 8B, A100 for 70B
5. Set environment variables (optional overrides):
   - `MAX_MODEL_LEN`: 8192
   - `GPU_MEMORY_UTILIZATION`: 0.9
   - `TENSOR_PARALLEL_SIZE`: 2 (for 70B on multi-GPU)

## Build Arguments

- `MODEL_NAME`: Hugging Face model ID (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `HF_TOKEN`: Your Hugging Face token for accessing gated models

## Example: Different Models

```powershell
# Mistral 7B
docker build `
  --build-arg MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" `
  --build-arg HF_TOKEN="your_token" `
  -t your-username/vllm-mistral-7b:latest `
  .

# Qwen 2.5 7B
docker build `
  --build-arg MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" `
  --build-arg HF_TOKEN="your_token" `
  -t your-username/vllm-qwen-7b:latest `
  .
```

## Benefits

✅ **Instant cold starts** - No model download delay
✅ **Consistent performance** - Model is cached in the image
✅ **Lower latency** - Worker starts immediately
✅ **Offline capable** - No dependency on HuggingFace at runtime

## Troubleshooting

**Build fails with authentication error:**
- Make sure your HF_TOKEN has access to the gated model
- Request access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

**Image too large:**
- Use Docker BuildKit for better layer caching
- Consider using quantized models (GPTQ, AWQ)

**RunPod endpoint fails to start:**
- Check Container Disk size (needs to fit the model)
- Verify GPU has enough VRAM for the model
- Check RunPod logs for detailed error messages

## Advanced: Multi-stage Build (Smaller Images)

You can optimize the image size by using a multi-stage build (not implemented yet, but possible).

## Notes

- The base image `runpod/worker-vllm:stable-cuda12.1.0` already includes vLLM and dependencies
- Model is stored in `/runpod-volume/` for compatibility with RunPod's volume system
- The OpenAI-compatible API starts automatically on port 8000

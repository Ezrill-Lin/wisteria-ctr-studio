# =========================================================
#  Pre-baked vLLM Image for RunPod Serverless
#  Author: ezrill
#  Description:
#     This image downloads the specified model once at build time
#     so the RunPod worker starts instantly (no Hugging Face download).
# =========================================================

# --- Base Image ---
FROM runpod/worker-v1-vllm:v2.9.6

# --- Environment Configuration ---
# You can replace MODEL_NAME with any Hugging Face model ID.
# For gated models, insert your HF token below.
ARG MODEL_NAME
ARG HF_TOKEN

ENV MODEL_NAME=${MODEL_NAME} \
    HF_TOKEN=${HF_TOKEN} \
    HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} \
    MAX_MODEL_LEN=8192 \
    GPU_MEMORY_UTILIZATION=0.9 \
    QUANTIZATION="" \
    TENSOR_PARALLEL_SIZE=1

# --- Pre-download the model into the image ---
# This downloads all model weights and tokenizer files during build
RUN huggingface-cli login --token ${HF_TOKEN} && \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_NAME}', local_dir='/runpod-volume/${MODEL_NAME}', token='${HF_TOKEN}')"

# --- Set model path ---
ENV MODEL_NAME=/runpod-volume/${MODEL_NAME}

# --- Entrypoint ---
# The base image handles starting the vLLM OpenAI-compatible server
# RunPod automatically runs it with the environment variables above


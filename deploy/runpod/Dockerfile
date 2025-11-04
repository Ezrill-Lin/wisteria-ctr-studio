FROM vllm/vllm-openai:latest

# Set working directory
WORKDIR /app

# Install additional dependencies if needed
RUN pip install --no-cache-dir \
    transformers \
    torch \
    accelerate

# Set environment variables for model caching
ENV HF_HOME=/data/cache
ENV TRANSFORMERS_CACHE=/data/cache
ENV HF_DATASETS_CACHE=/data/cache

# Create cache directory
RUN mkdir -p /data/cache

# Expose the port
EXPOSE 8000

# Default command for Llama 8B (can be overridden)
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model=meta-llama/Llama-3.1-8B-Instruct", \
     "--host=0.0.0.0", \
     "--port=8000", \
     "--trust-remote-code", \
     "--download-dir=/data/cache", \
     "--max-model-len=4096"]
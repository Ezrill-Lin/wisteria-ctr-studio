"""RunPod client base class.

This is the base RunPod client that provides basic configuration and shared functionality.
The SDK client inherits from this class and implements distributed inference logic.
"""

import os
from typing import Any, Dict, List, Optional

from .base_client import BaseLLMClient


class RunPodClient(BaseLLMClient):
    """Base RunPod client with model configuration and shared functionality."""
    
    def __init__(self, 
                 model: str = "llama3.1-8b",
                 api_key: Optional[str] = None,
                 timeout: int = 300):
        """Initialize RunPod client.
        
        Args:
            model: Model name (llama3.1-8b, llama3.1-70b).
            api_key: RunPod API key. If None, uses RUNPOD_API_KEY env var.
            timeout: Request timeout in seconds (default: 300).
        """
        super().__init__(model, api_key)
        
        self.timeout = timeout
        self.provider_name = "runpod"
        self.env_key_name = "RUNPOD_API_KEY"
        
        # Get API key for authentication
        self.runpod_api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not self.runpod_api_key:
            print("[WARNING] RunPod API key not found. Will use mock predictions.")
        
        # Model configurations for pod creation
        self.model_configs = {
            "llama3.1-8b": {
                "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "max_tokens": 10,
                "temperature": 0.1,
                "pod_template": "vllm/vllm-openai:latest",
                "gpu_type": "NVIDIA A40",
                "gpu_count": 1,
                "docker_args": "--host 0.0.0.0 --port 8000 --model meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --enforce-eager --gpu-memory-utilization 0.9 --max-model-len 8192 --max-num-seqs 128 --disable-log-stats"
            },
            "llama3.1-70b": {
                "name": "meta-llama/Llama-3.1-70B-Instruct",
                "model_name": "meta-llama/Llama-3.1-70B-Instruct", 
                "max_tokens": 10,
                "temperature": 0.1,
                "pod_template": "vllm/vllm-openai:latest",
                "gpu_type": "NVIDIA A100-SXM4-80GB",
                "gpu_count": 2,
                "docker_args": "--model meta-llama/Llama-3.1-70B-Instruct --host 0.0.0.0 --port 8000"
            }
        }
        
        if model not in self.model_configs:
            raise ValueError(f"Unsupported model: {model}. Available: {list(self.model_configs.keys())}")

        # Set to mock mode by default (SDK client will override)
        self.mode = "mock"
    
    def has_api_key(self) -> bool:
        """Check if RunPod API key is available."""
        return bool(self.runpod_api_key)
    
    def _calculate_max_tokens(self, batch_size: int) -> int:
        """Calculate max_tokens based on batch size."""
        return max(50, batch_size*10 + 20)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        config = self.model_configs[self.model]
        return {
            "provider": "runpod",
            "model": self.model,
            "model_name": config["model_name"],
            "deployment": "SDK",
            "has_api_key": self.has_api_key(),
            "mode": self.mode
        }
"""CTRPrediction (Click-Through Rate Prediction) Package

This package provides specialized tools for CTR (Click-Through Rate) prediction
using synthetic populations and LLM-based models. It includes multiple LLM 
provider clients and core prediction functionality for advertising effectiveness analysis.

Main Components:
- llm_click_model: Core LLM-based click prediction functionality with distributed support
- Client modules: Various LLM provider clients (OpenAI, DeepSeek, Gemini, RunPod, etc.)
- base_client: Abstract base class for LLM client implementations
- runpod_sdk_config: RunPod SDK configuration for automatic pod management

Features:
- Single-provider prediction: Use one LLM provider at a time
- Distributed prediction: Automatically load-balance across multiple providers (OpenAI, DeepSeek, Gemini)
  to avoid rate limits and improve throughput
"""

from .llm_click_model import (
    LLMClickPredictor, 
    DistributedLLMPredictor,
    predict_clicks_parallel,
    predict_clicks_distributed
)
from .base_client import BaseLLMClient
from .openai_client import OpenAIClient
from .deepseek_client import DeepSeekClient
from .gemini_client import GeminiClient
from .runpod_client import RunPodClient
from .runpod_sdk_config import RunPodSDKConfig

__version__ = "1.1.0"
__all__ = [
    "LLMClickPredictor", 
    "DistributedLLMPredictor",
    "predict_clicks_parallel",
    "predict_clicks_distributed",
    "BaseLLMClient", 
    "OpenAIClient",
    "DeepSeekClient",
    "GeminiClient",
    "RunPodClient", 
    "RunPodSDKConfig"
]
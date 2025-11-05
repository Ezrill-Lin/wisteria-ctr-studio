"""RunPod client for vLLM inference with HTTP pods.

Simple HTTP-only client for RunPod vLLM pods using OpenAI-compatible API.
"""

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

from .base_client import BaseLLMClient


class RunPodClient(BaseLLMClient):
    """RunPod client for LLM inference using HTTP pods.
    
    Clean HTTP-only implementation for RunPod vLLM pods with 
    OpenAI-compatible API endpoints.
    """
    
    def __init__(self, 
                 model: str = "llama-8b",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: int = 300):
        """Initialize RunPod client.
        
        Args:
            model: Model name (llama-8b, llama-70b).
            api_key: RunPod API key. If None, uses RUNPOD_API_KEY env var.
            base_url: HTTP base URL for vLLM endpoint.
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
        
        # Model configurations for HTTP endpoints
        self.model_configs = {
            "llama-8b": {
                "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "max_tokens": 10,
                "temperature": 0.1,
            },
            "llama-70b": {
                "model_name": "meta-llama/Llama-3.1-70B-Instruct", 
                "max_tokens": 10,
                "temperature": 0.1,
            }
        }
        
        if model not in self.model_configs:
            raise ValueError(f"Unsupported model: {model}. Available: {list(self.model_configs.keys())}")

        # Configure HTTP base URL
        self.http_base_url = base_url or self._get_http_base_url()

        # Determine mode: HTTP or mock
        self.mode = "http" if self.http_base_url else "mock"
        mode_name = {
            "http": "HTTP", 
            "mock": "mock (no config)"
        }[self.mode]
        print(f"[RunPod] {model} configured: {mode_name} mode")
    
    def _calculate_max_tokens(self, batch_size: int) -> int:
        """Calculate max_tokens based on batch size.
        
        Args:
            batch_size: Number of profiles in the batch
            
        Returns:
            Calculated max_tokens value
        """
        return max(50, batch_size*10 + 20)
    
    def _get_http_base_url(self) -> Optional[str]:
        """Get HTTP base URL for vLLM from env."""
        env_map = {
            "llama-8b": os.getenv("RUNPOD_LLAMA_8B_URL"),
            "llama-70b": os.getenv("RUNPOD_LLAMA_70B_URL"),
        }
        return os.getenv("RUNPOD_BASE_URL") or env_map.get(self.model)
    
    def has_api_key(self) -> bool:
        """Check if RunPod API key is available."""
        return bool(self.runpod_api_key)
    
    async def predict_chunk_async(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
        """Predict clicks for a chunk of profiles asynchronously."""
        if self.mode == "http":
            return await self._predict_http_async(ad_text, profiles, ad_platform)
        
        # No valid configuration
        return self._fallback_to_mock(ad_text, profiles, "RunPod HTTP not configured")
    
    def predict_chunk(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
        """Predict clicks for a chunk of profiles synchronously."""
        if self.mode == "http":
            return self._predict_http_sync(ad_text, profiles, ad_platform)
        
        # No valid configuration
        return self._fallback_to_mock(ad_text, profiles, "RunPod HTTP not configured")
    
    # ---- HTTP (OpenAI-compatible) helpers ----
    def _create_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You are a precise decision engine that outputs strict JSON."},
            {"role": "user", "content": prompt},
        ]

    def _predict_http_sync(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str) -> List[int]:
        if not self.http_base_url:
            return self._fallback_to_mock(ad_text, profiles, "RunPod HTTP base URL not configured")
        prompt = self._build_prompt(ad_text, profiles, ad_platform)
        try:
            from openai import OpenAI
            # Calculate max_tokens based on batch size
            max_tokens = self._calculate_max_tokens(len(profiles))
            # Use RunPod API key for authentication
            api_key = self.runpod_api_key or os.getenv("RUNPOD_API_KEY", "dummy")
            client = OpenAI(api_key=api_key, base_url=self.http_base_url, timeout=self.timeout)
            resp = client.chat.completions.create(
                model=self.model_configs[self.model]["model_name"],
                messages=self._create_messages(prompt),
                temperature=self.model_configs[self.model]["temperature"],
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            
        except Exception as e:
            return self._fallback_to_mock(ad_text, profiles, f"RunPod HTTP sync error ({type(e).__name__}: {str(e)})")
        return self._parse_and_validate_response(content, profiles, ad_text, "RunPod HTTP")

    async def _predict_http_async(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str) -> List[int]:
        if not self.http_base_url:
            return self._fallback_to_mock(ad_text, profiles, "RunPod HTTP base URL not configured")
        prompt = self._build_prompt(ad_text, profiles, ad_platform)
        try:
            from openai import AsyncOpenAI
            # Calculate max_tokens based on batch size
            max_tokens = self._calculate_max_tokens(len(profiles))
            # Use RunPod API key for authentication
            api_key = self.runpod_api_key or os.getenv("RUNPOD_API_KEY", "dummy")
            client = AsyncOpenAI(api_key=api_key, base_url=self.http_base_url, timeout=self.timeout)
            resp = await client.chat.completions.create(
                model=self.model_configs[self.model]["model_name"],
                messages=self._create_messages(prompt),
                temperature=self.model_configs[self.model]["temperature"],
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            
        except Exception as e:
            return self._fallback_to_mock(ad_text, profiles, f"RunPod HTTP async error ({type(e).__name__}: {str(e)})")
        return self._parse_and_validate_response(content, profiles, ad_text, "RunPod HTTP")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        config = self.model_configs[self.model]
        deployment_map = {
            "http": "HTTP",
            "mock": "mock (no config)"
        }
        return {
            "provider": "runpod",
            "model": self.model,
            "model_name": config["model_name"],
            "deployment": deployment_map.get(self.mode, self.mode),
            "base_url": self.http_base_url,
            "has_api_key": self.has_api_key(),
            "mode": self.mode
        }
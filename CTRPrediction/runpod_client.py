"""RunPod client for vLLM inference with Llama models.

Supports two modes:
- HTTP (OpenAI-compatible) if a base URL is provided via env (preferred)
- Serverless Jobs API using endpoint IDs (fallback)
"""

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional
import aiohttp
import requests

from .base_client import BaseLLMClient


class RunPodClient(BaseLLMClient):
    """RunPod client for LLM inference using serverless endpoints.
    
    Simplified to only support serverless endpoints for cost-effective
    auto-scaling without idle costs.
    """
    
    def __init__(self, 
                 model: str = "llama-8b",
                 api_key: Optional[str] = None,
                 endpoint_id: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: int = 300):
        """Initialize RunPod client.
        
        Args:
            model: Model name (llama-8b, llama-70b).
            api_key: RunPod API key. If None, uses RUNPOD_API_KEY env var.
            endpoint_id: RunPod endpoint ID for serverless.
            timeout: Request timeout in seconds.
        """
        super().__init__(model, api_key)
        
        self.timeout = timeout
        self.provider_name = "runpod"
        self.env_key_name = "RUNPOD_API_KEY"
        
        # Get API key
        self.runpod_api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not self.runpod_api_key:
            print("⚠️ RunPod API key not found. Will use mock predictions.")
        
        # Model configurations
        self.model_configs = {
            "llama-8b": {
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "max_tokens": 10,
                "temperature": 0.1,
                "gpu_type": "RTX4090",
                "cost_per_hour": 0.39,
                "endpoint_var": "RUNPOD_LLAMA_8B_ENDPOINT"
            },
            "llama-70b": {
                "model_name": "meta-llama/Llama-3.1-70B-Instruct", 
                "max_tokens": 10,
                "temperature": 0.1,
                "gpu_type": "A100",
                "cost_per_hour": 1.89,
                "endpoint_var": "RUNPOD_LLAMA_70B_ENDPOINT"
            }
        }
        
        if model not in self.model_configs:
            raise ValueError(f"Unsupported model: {model}. Available: {list(self.model_configs.keys())}")

        # Configure endpoints
        self.endpoint_id = endpoint_id or self._get_endpoint_id()
        self.http_base_url = base_url or self._get_http_base_url()
        self.base_url = self._configure_jobs_url()

        # Decide mode (prefer HTTP if base URL provided)
        self.mode = "http" if self.http_base_url else "jobs"
        print(f"🏃 RunPod {model} configured: {'HTTP (OpenAI)' if self.mode == 'http' else 'serverless jobs'} mode")
    
    def _get_endpoint_id(self) -> Optional[str]:
        """Get endpoint ID from environment variables."""
        config = self.model_configs[self.model]
        endpoint_id = os.getenv(config["endpoint_var"])
        if not endpoint_id:
            print(f"⚠️ {config['endpoint_var']} not set. Set up endpoints first.")
        return endpoint_id
    
    def _configure_jobs_url(self) -> str:
        """Configure the serverless jobs API base URL."""
        if not self.endpoint_id:
            return "mock://runpod-serverless"
        return f"https://api.runpod.ai/v2/{self.endpoint_id}"

    def _get_http_base_url(self) -> Optional[str]:
        """Get OpenAI-compatible HTTP base URL for vLLM from env, if any."""
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
        # HTTP mode (OpenAI-compatible)
        if self.mode == "http":
            return await self._predict_http_async(ad_text, profiles, ad_platform)

        if not self.has_api_key() or "mock://" in self.base_url:
            # Fallback to mock prediction
            from .base_client import _mock_predict
            return _mock_predict(ad_text, profiles)
        
        try:
            prompt = self._build_prompt(ad_text, profiles, ad_platform)
            
            response_text = await self._predict_serverless(prompt)
            
            return self._parse_and_validate_response(response_text, profiles, ad_text, "RunPod Jobs")
            
        except Exception as e:
            print(f"[RunPod Error] {str(e)}")
            # Fallback to mock prediction
            from .base_client import _mock_predict
            return _mock_predict(ad_text, profiles)
    
    def predict_chunk(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
        """Predict clicks for a chunk of profiles synchronously."""
        # HTTP mode (OpenAI-compatible)
        if self.mode == "http":
            return self._predict_http_sync(ad_text, profiles, ad_platform)

        if not self.has_api_key() or "mock://" in self.base_url:
            # Fallback to mock prediction
            from .base_client import _mock_predict
            return _mock_predict(ad_text, profiles)
        
        try:
            prompt = self._build_prompt(ad_text, profiles, ad_platform)
            
            response_text = self._predict_serverless_sync(prompt)
            
            return self._parse_and_validate_response(response_text, profiles, ad_text, "RunPod Jobs")
            
        except Exception as e:
            print(f"[RunPod Error] {str(e)}")
            # Fallback to mock prediction
            from .base_client import _mock_predict
            return _mock_predict(ad_text, profiles)
    
    async def _predict_serverless(self, prompt: str) -> str:
        """Make async request to RunPod serverless endpoint."""
        config = self.model_configs[self.model]
        
        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": config["max_tokens"],
                "temperature": config["temperature"],
                "model": config["model_name"]
            }
        }
        
        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(
                f"{self.base_url}/runsync",
                json=payload,
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"RunPod serverless failed with status {response.status}: {error_text}")
                
                result = await response.json()
                
                if result.get("status") == "COMPLETED":
                    output = result.get("output", {})
                    if isinstance(output, dict):
                        return output.get("text", "")
                    return str(output)
                else:
                    raise Exception(f"RunPod job failed: {result}")
    
    def _predict_serverless_sync(self, prompt: str) -> str:
        """Make sync request to RunPod serverless endpoint."""
        config = self.model_configs[self.model]
        
        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": config["max_tokens"],
                "temperature": config["temperature"],
                "model": config["model_name"]
            }
        }
        
        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{self.base_url}/runsync",
            json=payload,
            headers=headers,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"RunPod serverless failed with status {response.status_code}: {response.text}")
        
        result = response.json()
        
        if result.get("status") == "COMPLETED":
            output = result.get("output", {})
            if isinstance(output, dict):
                return output.get("text", "")
            return str(output)
        else:
            raise Exception(f"RunPod job failed: {result}")

    # ---- HTTP (OpenAI-compatible) helpers ----
    def _create_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You are a precise decision engine that outputs strict JSON."},
            {"role": "user", "content": prompt},
        ]

    def _predict_http_sync(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str) -> List[int]:
        if not self.http_base_url:
            from .base_client import _mock_predict
            return _mock_predict(ad_text, profiles)
        prompt = self._build_prompt(ad_text, profiles, ad_platform)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("RUNPOD_HTTP_API_KEY", "dummy"), base_url=self.http_base_url)
            resp = client.chat.completions.create(
                model=self.model_configs[self.model]["model_name"],
                messages=self._create_messages(prompt),
                temperature=self.model_configs[self.model]["temperature"],
                max_tokens=self.model_configs[self.model]["max_tokens"],
            )
            content = resp.choices[0].message.content
        except Exception:
            from .base_client import _mock_predict
            return _mock_predict(ad_text, profiles)
        return self._parse_and_validate_response(content, profiles, ad_text, "RunPod HTTP")

    async def _predict_http_async(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str) -> List[int]:
        if not self.http_base_url:
            from .base_client import _mock_predict
            return _mock_predict(ad_text, profiles)
        prompt = self._build_prompt(ad_text, profiles, ad_platform)
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.getenv("RUNPOD_HTTP_API_KEY", "dummy"), base_url=self.http_base_url)
            resp = await client.chat.completions.create(
                model=self.model_configs[self.model]["model_name"],
                messages=self._create_messages(prompt),
                temperature=self.model_configs[self.model]["temperature"],
                max_tokens=self.model_configs[self.model]["max_tokens"],
            )
            content = resp.choices[0].message.content
        except Exception:
            from .base_client import _mock_predict
            return _mock_predict(ad_text, profiles)
        return self._parse_and_validate_response(content, profiles, ad_text, "RunPod HTTP")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        config = self.model_configs[self.model]
        return {
            "provider": "runpod",
            "model": self.model,
            "model_name": config["model_name"],
            "gpu_type": config["gpu_type"],
            "deployment": "http" if self.mode == "http" else "serverless",
            "cost_per_hour": config["cost_per_hour"],
            "endpoint_id": self.endpoint_id,
            "base_url": self.http_base_url or self.base_url,
            "has_api_key": self.has_api_key()
        }
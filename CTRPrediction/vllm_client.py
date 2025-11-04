"""vLLM client for distributed LLM inference with Llama models."""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
import aiohttp

from .base_client import BaseLLMClient


class VLLMClient(BaseLLMClient):
    """Client for vLLM inference server with Llama model support.
    
    Supports multiple Llama models with automatic model selection and
    proper GPU resource allocation via GKE routing.
    """
    
    def __init__(self, 
                 model: str = "llama-8b",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: int = 300):
        """Initialize vLLM client.
        
        Args:
            model: Model name (llama-8b, llama-70b).
            api_key: Optional API key for secured vLLM deployments.
            base_url: vLLM server base URL. If None, uses environment variable.
            timeout: Request timeout in seconds.
        """
        super().__init__(model, api_key)
        
        # Default to environment variable or local development
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        self.timeout = timeout
        self.provider_name = "vllm"
        self.env_key_name = "VLLM_API_KEY"
        # Model configurations
        self.model_configs = {
            "llama-8b": {
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "max_tokens": 10,
                "temperature": 0.1,
                "description": "Llama 3.1 8B - Fast, cost-effective"
            },
            "llama-70b": {
                "model_name": "meta-llama/Llama-3.1-70B-Instruct", 
                "max_tokens": 10,
                "temperature": 0.1,
                "description": "Llama 3.1 70B - High accuracy"
            }
        }
        
        if self.model not in self.model_configs:
            raise ValueError(f"Unsupported model: {self.model}. Available: {list(self.model_configs.keys())}")
    
    def _prepare_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Prepare messages in chat format for Llama models."""
        return [
            {
                "role": "user",
                "content": prompt
            }
        ]
    
    async def _make_request(self, messages: List[Dict[str, str]]) -> str:
        """Make async request to vLLM server."""
        config = self.model_configs[self.model]
        
        # Construct the proper endpoint for model routing
        endpoint_path = f"/v1/{self.model}/chat/completions"
        
        payload = {
            "model": self.model,  # Use the simplified model name for routing
            "messages": messages,
            "max_tokens": config["max_tokens"],
            "temperature": config["temperature"],
            "stream": False
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            try:
                async with session.post(
                    f"{self.base_url}{endpoint_path}",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"vLLM request failed with status {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "choices" not in result or not result["choices"]:
                        raise Exception("No choices in vLLM response")
                    
                    return result["choices"][0]["message"]["content"].strip()
                    
            except asyncio.TimeoutError:
                raise Exception(f"vLLM request timed out after {self.timeout}s")
            except Exception as e:
                raise Exception(f"vLLM request failed: {str(e)}")
    
    def _make_request_sync(self, messages: List[Dict[str, str]]) -> str:
        """Make synchronous request to vLLM server."""
        import requests
        
        config = self.model_configs[self.model]
        
        # Construct the proper endpoint for model routing
        endpoint_path = f"/v1/{self.model}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": config["max_tokens"],
            "temperature": config["temperature"],
            "stream": False
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(
                f"{self.base_url}{endpoint_path}",
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"vLLM request failed with status {response.status_code}: {response.text}")
            
            result = response.json()
            
            if "choices" not in result or not result["choices"]:
                raise Exception("No choices in vLLM response")
            
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.Timeout:
            raise Exception(f"vLLM request timed out after {self.timeout}s")
        except Exception as e:
            raise Exception(f"vLLM request failed: {str(e)}")
    
    async def predict_chunk_async(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
        """Predict clicks for a chunk of profiles asynchronously."""
        try:
            prompt = self._build_prompt(ad_text, profiles, ad_platform)
            messages = self._prepare_messages(prompt)
            response_text = await self._make_request(messages)
            return self._parse_response(response_text, len(profiles))
        except Exception as e:
            print(f"[vLLM Error] {str(e)}")
            # Fallback to mock prediction
            from .base_client import _mock_predict
            return _mock_predict(ad_text, profiles)
    
    def predict_chunk(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
        """Predict clicks for a chunk of profiles synchronously."""
        try:
            prompt = self._build_prompt(ad_text, profiles, ad_platform)
            messages = self._prepare_messages(prompt)
            response_text = self._make_request_sync(messages)
            return self._parse_response(response_text, len(profiles))
        except Exception as e:
            print(f"[vLLM Error] {str(e)}")
            # Fallback to mock prediction
            from .base_client import _mock_predict
            return _mock_predict(ad_text, profiles)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        config = self.model_configs[self.model]
        return {
            "provider": "vllm",
            "model": self.model,
            "full_model_name": config["model_name"],
            "description": config["description"],
            "base_url": self.base_url
        }
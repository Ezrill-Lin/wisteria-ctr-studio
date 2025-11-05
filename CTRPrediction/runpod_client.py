"""RunPod client base class with URL validation.

This is the base RunPod client that provides URL validation and basic configuration.
HTTP and SDK clients inherit from this class and implement the actual prediction logic.
"""

import json
import os
import sys
import time
import requests
from typing import Any, Dict, List, Optional

from .base_client import BaseLLMClient


class RunPodClient(BaseLLMClient):
    """Base RunPod client with URL validation and configuration.
    
    This provides common functionality for RunPod clients including:
    - URL validation with interactive prompts
    - Model configuration management
    - RunPod SDK initialization
    - Pod management methods
    """
    
    def __init__(self, 
                 model: str = "llama3.1-8b",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: int = 300,
                 auto_create_pod: bool = False,
                 pod_config: Optional[Dict[str, Any]] = None):
        """Initialize RunPod client.
        
        Args:
            model: Model name (llama3.1-8b, llama3.1-70b).
            api_key: RunPod API key. If None, uses RUNPOD_API_KEY env var.
            base_url: HTTP base URL for vLLM endpoint.
            timeout: Request timeout in seconds (default: 300).
            auto_create_pod: Whether to automatically create a pod if needed.
            pod_config: Custom pod configuration for automatic creation.
        """
        super().__init__(model, api_key)
        
        self.timeout = timeout
        self.provider_name = "runpod"
        self.env_key_name = "RUNPOD_API_KEY"
        self.auto_create_pod = auto_create_pod
        self.pod_config = pod_config or {}
        
        # Get API key for authentication
        self.runpod_api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not self.runpod_api_key:
            print("[WARNING] RunPod API key not found. Will use mock predictions.")
        
        # Model configurations
        self.model_configs = {
            "llama3.1-8b": {
                "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "max_tokens": 10,
                "temperature": 0.1,
            },
            "llama3.1-70b": {
                "model_name": "meta-llama/Llama-3.1-70B-Instruct", 
                "max_tokens": 10,
                "temperature": 0.1,
            }
        }
        
        if model not in self.model_configs:
            raise ValueError(f"Unsupported model: {model}. Available: {list(self.model_configs.keys())}")

        # Configure HTTP base URL
        self.http_base_url = base_url or self._get_http_base_url()
        
        # Set mode based on available configuration
        if self.http_base_url:
            # Validate URL and set mode
            if self._validate_base_url(self.http_base_url):
                self.mode = "http"
                print(f"[RunPod] Configured for HTTP mode: {self.http_base_url}")
            else:
                self.mode = "mock"
                print(f"[RunPod] HTTP URL validation failed - using mock mode")
        else:
            self.mode = "mock"
            print(f"[RunPod] No HTTP URL configured - using mock mode")
    
    def _get_http_base_url(self) -> Optional[str]:
        """Get HTTP base URL for vLLM from env."""
        env_map = {
            "llama3.1-8b": os.getenv("RUNPOD_LLAMA_8B_URL"),
            "llama3.1-70b": os.getenv("RUNPOD_LLAMA_70B_URL"),
        }
        return os.getenv("RUNPOD_BASE_URL") or env_map.get(self.model)
    
    def has_api_key(self) -> bool:
        """Check if RunPod API key is available."""
        return bool(self.runpod_api_key)
    
    def _calculate_max_tokens(self, batch_size: int) -> int:
        """Calculate max_tokens based on batch size."""
        return max(50, batch_size*10 + 20)

    def _validate_base_url(self, base_url: str, timeout: int = 10) -> bool:
        """Validate if the base URL is accessible and running vLLM.
        
        Args:
            base_url: The base URL to validate
            timeout: Request timeout in seconds
            
        Returns:
            True if URL is valid and accessible, False otherwise
        """
        if not base_url:
            return False
            
        try:
            # Try to access the models endpoint to verify vLLM is running
            models_url = base_url.rstrip('/') + '/models'
            response = requests.get(models_url, timeout=timeout)
            
            if response.status_code == 200:
                # Check if it's a valid OpenAI-compatible response
                try:
                    data = response.json()
                    if 'data' in data and isinstance(data['data'], list):
                        print(f"[RunPod] URL validation successful: {base_url}")
                        return True
                except (json.JSONDecodeError, KeyError):
                    pass
            
            print(f"[RunPod] URL validation failed: {base_url} (Status: {response.status_code})")
            return False
            
        except requests.RequestException as e:
            print(f"[RunPod] URL validation failed: {base_url} (Error: {str(e)})")
            return False
        except Exception as e:
            print(f"[RunPod] URL validation failed: {base_url} (Unexpected error: {str(e)})")
            return False

    def validate_and_update_url(self, new_url: Optional[str] = None) -> bool:
        """Validate current URL or set and validate a new URL.
        
        Args:
            new_url: Optional new URL to validate and set. If None, validates current URL.
            
        Returns:
            True if URL is valid and accessible, False otherwise.
        """
        url_to_validate = new_url or self.http_base_url
        
        if not url_to_validate:
            print("[RunPod] No URL to validate")
            return False
        
        print(f"[RunPod] Validating URL: {url_to_validate}")
        is_valid = self._validate_base_url(url_to_validate)
        
        if is_valid and new_url:
            self.http_base_url = new_url
            self.mode = "http"
            print(f"[RunPod] URL updated and validated successfully")
        elif is_valid:
            print(f"[RunPod] Current URL is valid")
        else:
            print(f"[RunPod] URL validation failed")
            self.mode = "mock"
        
        return is_valid

    def _create_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Create OpenAI-compatible message structure."""
        return [
            {"role": "system", "content": "You are a precise decision engine that outputs strict JSON."},
            {"role": "user", "content": prompt},
        ]

    async def predict_chunk_async(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
        """Predict clicks for a chunk of profiles asynchronously."""
        # If we have a valid HTTP URL, use HTTP mode
        if self.http_base_url and self._validate_base_url(self.http_base_url, timeout=5):
            self.mode = "http"
            return await self._predict_http_async(ad_text, profiles, ad_platform)
        
        # Fallback to mock
        return self._fallback_to_mock(ad_text, profiles, "RunPod not configured - using mock predictions")
    
    def predict_chunk(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
        """Predict clicks for a chunk of profiles synchronously."""
        # If we have a valid HTTP URL, use HTTP mode
        if self.http_base_url and self._validate_base_url(self.http_base_url, timeout=5):
            self.mode = "http"
            return self._predict_http_sync(ad_text, profiles, ad_platform)
        
        # Fallback to mock
        return self._fallback_to_mock(ad_text, profiles, "RunPod not configured - using mock predictions")

    async def _predict_http_async(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str) -> List[int]:
        """HTTP prediction for pods (async)."""
        if not self.http_base_url:
            return self._fallback_to_mock(ad_text, profiles, "No HTTP URL available")
            
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
        
        return self._parse_and_validate_response(content, profiles, ad_text, "RunPod")

    def _predict_http_sync(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str) -> List[int]:
        """HTTP prediction for pods (sync)."""
        if not self.http_base_url:
            return self._fallback_to_mock(ad_text, profiles, "No HTTP URL available")
            
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
        
        return self._parse_and_validate_response(content, profiles, ad_text, "RunPod")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        config = self.model_configs[self.model]
        return {
            "provider": "runpod",
            "model": self.model,
            "model_name": config["model_name"],
            "deployment": "HTTP",
            "base_url": self.http_base_url,
            "has_api_key": self.has_api_key(),
            "mode": self.mode
        }
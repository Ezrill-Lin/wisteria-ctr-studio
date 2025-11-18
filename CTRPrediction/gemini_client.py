"""Google Gemini client implementation for LLM click prediction."""

import os
from typing import Any, Dict, List

from .base_client import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """Google Gemini client for click prediction."""
    
    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = None):
        """Initialize Gemini client.
        
        Args:
            model: Model name to use (gemini-1.5-flash, gemini-1.5-pro).
            api_key: API key override (else read from GEMINI_API_KEY env var).
        """
        super().__init__(model, api_key)
        self.provider_name = "gemini"
        self.env_key_name = "GEMINI_API_KEY"

    def _get_client(self):
        """Get synchronous Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key or os.getenv(self.env_key_name))
            return genai.GenerativeModel(self.model)
        except Exception as e:
            raise ImportError(f"Gemini import failed: {e}")
    
    async def _get_async_client(self):
        """Get asynchronous Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key or os.getenv(self.env_key_name))
            return genai.GenerativeModel(self.model)
        except Exception as e:
            raise ImportError(f"Gemini import failed: {e}")

    def _create_prompt(self, prompt: str) -> str:
        """Create prompt format for Gemini API."""
        system_instruction = "You are a precise decision engine that outputs strict JSON."
        return f"{system_instruction}\n\n{prompt}"

    def predict_chunk(self, ad_text: str, chunk: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
        """Synchronous prediction for a chunk of profiles."""
        if not self.has_api_key():
            return self._fallback_to_mock(ad_text, chunk, "Gemini API key missing")
        
        try:
            client = self._get_client()
        except ImportError:
            return self._fallback_to_mock(ad_text, chunk, "Gemini import failed")
        except Exception:
            return self._fallback_to_mock(ad_text, chunk, "Gemini client init failed")
        
        prompt = self._build_prompt(ad_text, chunk, ad_platform)
        full_prompt = self._create_prompt(prompt)
        
        try:
            response = client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 1000,
                }
            )
            content = response.text
        except Exception as e:
            return self._fallback_to_mock(ad_text, chunk, f"Gemini API call failed ({type(e).__name__}: {str(e)})")
        
        return self._parse_and_validate_response(content, chunk, ad_text, "Gemini")
    
    async def predict_chunk_async(self, ad_text: str, chunk: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
        """Asynchronous prediction for a chunk of profiles."""
        if not self.has_api_key():
            return self._fallback_to_mock(ad_text, chunk, "Gemini API key missing")
        
        try:
            client = await self._get_async_client()
        except ImportError:
            return self._fallback_to_mock(ad_text, chunk, "Async Gemini import failed")
        except Exception:
            return self._fallback_to_mock(ad_text, chunk, "Async Gemini client init failed")
        
        prompt = self._build_prompt(ad_text, chunk, ad_platform)
        full_prompt = self._create_prompt(prompt)
        
        try:
            response = await client.generate_content_async(
                full_prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 1000,
                }
            )
            content = response.text
        except Exception as e:
            return self._fallback_to_mock(ad_text, chunk, f"Gemini async API call failed ({type(e).__name__}: {str(e)})")
        
        return self._parse_and_validate_response(content, chunk, ad_text, "Gemini")

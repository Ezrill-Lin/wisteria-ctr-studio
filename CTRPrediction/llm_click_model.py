import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .openai_client import OpenAIClient
from .deepseek_client import DeepSeekClient
from .runpod_client import RunPodClient
from .base_client import _print_fallback, _mock_predict
# from .template_client import TemplateClient  # TODO: Add other clients as needed


def _chunked(seq: List[Any], n: int) -> Iterable[List[Any]]:
    """Yield successive chunks of size ``n`` from ``seq``.

    Args:
        seq: Input sequence to split.
        n: Chunk size (> 0).

    Yields:
        Consecutive sublists from ``seq`` of length up to ``n``.
    """
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# Client registry for easy extensibility
CLIENT_REGISTRY = {
    "openai": OpenAIClient,
    "deepseek": DeepSeekClient,
    "runpod": RunPodClient,  # RunPod for distributed inference with HTTP pods
    "vllm": RunPodClient,    # vLLM provider alias for RunPod HTTP client
    # "template": TemplateClient,  # TODO: Add other clients here
}


@dataclass
class LLMClickPredictor:
    """Modular batched LLM-based click predictor with support for multiple providers.

    Uses a registry-based client system for easy extensibility. New providers can
    be added by creating a client class that inherits from BaseLLMClient and 
    registering it in CLIENT_REGISTRY.

    Attributes:
        provider: LLM provider identifier (e.g., "openai", "deepseek", "vllm").
        model: Model name for the provider.
        batch_size: Number of profiles per request.
        use_mock: If True, always use the mock predictor.
        use_async: If True, use async parallel processing; if False, use sequential processing.
        api_key: Optional API key override (else read from env).
        runpod_base_url: RunPod HTTP base URL for vLLM endpoints.
    """
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    batch_size: int = 50
    use_mock: bool = False
    use_async: bool = True
    api_key: Optional[str] = None
    runpod_base_url: Optional[str] = None

    def __post_init__(self):
        """Initialize the appropriate client after dataclass creation.

        Note: Must set self._client; do not return a different object here.
        """
        if self.provider not in CLIENT_REGISTRY:
            raise ValueError(f"Unknown provider '{self.provider}'. Available providers: {list(CLIENT_REGISTRY.keys())}")

        client_class = CLIENT_REGISTRY[self.provider]

        # Handle vLLM/RunPod-specific initialization
        if self.provider in ["runpod", "vllm"]:
            self._client = RunPodClient(
                model=self.model,
                api_key=self.api_key,
                base_url=self.runpod_base_url
            )
        else:
            # Standard initialization for other providers
            self._client = client_class(
                model=self.model,
                api_key=self.api_key
            )

    def _use_real(self) -> bool:
        """Return True if real API calls should be used.

        Considers ``use_mock`` and whether the client is configured for real API calls.
        """
        if self.use_mock:
            return False
        return self._client.has_api_key()

    async def predict_clicks_async(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
        """Async version: Predict binary clicks for all profiles with parallel processing.

        Handles batching and runs API calls in parallel for better performance.

        Args:
            ad_text: Advertisement copy to evaluate.
            profiles: Profiles to score.
            ad_platform: Platform where the ad is shown (facebook, tiktok, amazon).

        Returns:
            List of 0/1 integers aligned with ``profiles`` order.
        """
        clicks: List[int] = []
        if not profiles:
            return clicks
            
        if not self._use_real():
            # Print a reasoned fallback only if not explicitly in mock mode
            if not self.use_mock:
                _print_fallback(f"{self.provider} API not configured; using mock for all chunks.")
            for chunk in _chunked(profiles, self.batch_size):
                clicks.extend(_mock_predict(ad_text, chunk))
            return clicks

        # Real calls in parallel batches
        chunks = list(_chunked(profiles, self.batch_size))
        tasks = [
            self._client.predict_chunk_async(ad_text, chunk, ad_platform)
            for chunk in chunks
        ]
        results = await asyncio.gather(*tasks)
        for result in results:
            clicks.extend(result)
        return clicks

    def predict_clicks(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
        """Predict binary clicks for all profiles.

        Handles batching and chooses between real API calls and the mock
        implementation based on configuration.

        Args:
            ad_text: Advertisement copy to evaluate.
            profiles: Profiles to score.
            ad_platform: Platform where the ad is shown (facebook, tiktok, amazon).

        Returns:
            List of 0/1 integers aligned with ``profiles`` order.
        """
        clicks: List[int] = []
        if not profiles:
            return clicks
            
        if not self._use_real():
            # Print a reasoned fallback only if not explicitly in mock mode
            if not self.use_mock:
                _print_fallback(f"{self.provider} API not configured; using mock for all chunks.")
            for chunk in _chunked(profiles, self.batch_size):
                clicks.extend(_mock_predict(ad_text, chunk))
            return clicks

        # Real calls in batches
        # Use synchronous sequential processing only
        for chunk in _chunked(profiles, self.batch_size):
            clicks.extend(self._client.predict_chunk(ad_text, chunk, ad_platform))
        return clicks


async def predict_clicks_parallel(predictor: LLMClickPredictor, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
    """Convenience function to run async predictions in parallel.
    
    This is useful when you want to explicitly use async processing from 
    an already async context without the automatic asyncio.run() wrapper.
    
    Args:
        predictor: The LLMClickPredictor instance.
        ad_text: Advertisement copy to evaluate.
        profiles: Profiles to score.
        ad_platform: Platform where the ad is shown (facebook, tiktok, amazon).
        
    Returns:
        List of 0/1 integers aligned with ``profiles`` order.
    """
    return await predictor.predict_clicks_async(ad_text, profiles, ad_platform)


__all__ = ["LLMClickPredictor", "predict_clicks_parallel"]

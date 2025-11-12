import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from tqdm import tqdm

from .openai_client import OpenAIClient
from .deepseek_client import DeepSeekClient
from .runpod_sdk_config import RunPodSDKConfig
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
    "vllm": RunPodSDKConfig,     # vLLM distributed inference via RunPod SDK
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
        api_key: Optional API key override (else read from env).
        profiles_per_pod: For vllm provider, number of profiles per pod for distributed inference.
        population_size: Total population size for optimal pod calculation (for vllm provider).
    """
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    batch_size: int = 50
    use_mock: bool = False
    api_key: Optional[str] = None
    profiles_per_pod: int = 5000
    population_size: Optional[int] = None

    def __post_init__(self):
        """Initialize the appropriate client after dataclass creation.

        Note: Must set self._client; do not return a different object here.
        """
        if self.provider not in CLIENT_REGISTRY:
            raise ValueError(f"Unknown provider '{self.provider}'. Available providers: {list(CLIENT_REGISTRY.keys())}")

        client_class = CLIENT_REGISTRY[self.provider]

        # Handle vLLM distributed inference via RunPod SDK
        if self.provider == "vllm":
            self._client = RunPodSDKConfig(
                model=self.model,
                api_key=self.api_key,
                profiles_per_pod=self.profiles_per_pod,
                population_size=self.population_size
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

        # Real calls in parallel batches with progress tracking
        chunks = list(_chunked(profiles, self.batch_size))
        
        # Create async wrapper to track progress
        async def predict_with_progress(chunk, pbar):
            result = await self._client.predict_chunk_async(ad_text, chunk, ad_platform)
            pbar.update(1)
            return result
        
        # Initialize progress bar
        with tqdm(total=len(chunks), desc=f"Processing batches ({self.provider})", unit="batch") as pbar:
            tasks = [predict_with_progress(chunk, pbar) for chunk in chunks]
            results = await asyncio.gather(*tasks)
        
        for result in results:
            clicks.extend(result)
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

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
from tqdm import tqdm

from .openai_client import OpenAIClient
from .deepseek_client import DeepSeekClient
from .gemini_client import GeminiClient
from .runpod_sdk_config import RunPodSDKConfig
from .base_client import _print_fallback, _mock_predict, BaseLLMClient

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
    "gemini": GeminiClient,
    "vllm": RunPodSDKConfig,     # vLLM distributed inference via RunPod SDK
    # "template": TemplateClient,  # TODO: Add other clients here
}


@dataclass
class DistributedLLMPredictor:
    """Distributed load-balanced LLM predictor that uses multiple providers simultaneously.
    
    This predictor automatically distributes batches across multiple LLM providers
    (OpenAI, DeepSeek, Gemini) to avoid rate limits and improve throughput.
    
    Attributes:
        batch_size: Number of profiles per request.
        use_mock: If True, always use the mock predictor.
        providers: List of provider names to use (default: ["openai", "deepseek", "gemini"]).
        models: Dict mapping provider names to model names (optional, uses defaults if not specified).
        api_keys: Dict mapping provider names to API keys (optional, uses env vars if not specified).
    """
    batch_size: int = 50
    use_mock: bool = False
    providers: List[str] = field(default_factory=lambda: ["openai", "deepseek", "gemini"])
    models: Optional[Dict[str, str]] = None
    api_keys: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Initialize all available clients for distributed processing."""
        self._clients: List[BaseLLMClient] = []
        self._provider_names: List[str] = []
        
        # Default models for each provider
        default_models = {
            "openai": "gpt-4o-mini",
            "deepseek": "deepseek-chat",
            "gemini": "gemini-1.5-flash",
        }
        
        # Initialize clients for each provider
        for provider in self.providers:
            if provider == "vllm":
                # Skip vLLM for distributed mode (too complex for load balancing)
                print(f"[Distributed] Skipping vLLM provider (not suitable for distributed mode)")
                continue
                
            if provider not in CLIENT_REGISTRY:
                print(f"[Distributed] Unknown provider '{provider}', skipping")
                continue
            
            # Get model and API key
            model = self.models.get(provider) if self.models else default_models.get(provider, "default")
            api_key = self.api_keys.get(provider) if self.api_keys else None
            
            # Create client
            client_class = CLIENT_REGISTRY[provider]
            try:
                client = client_class(model=model, api_key=api_key)
                
                # Only add client if it has an API key
                if client.has_api_key():
                    self._clients.append(client)
                    self._provider_names.append(provider)
                    print(f"[Distributed] Initialized {provider} client with model {model}")
                else:
                    print(f"[Distributed] Skipping {provider} (no API key found)")
            except Exception as e:
                print(f"[Distributed] Failed to initialize {provider}: {e}")
        
        if not self._clients:
            print("[Distributed] WARNING: No clients initialized, will use mock predictions")

    def _use_real(self) -> bool:
        """Return True if real API calls should be used."""
        return not self.use_mock and len(self._clients) > 0

    async def predict_clicks_async(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
        """Async version: Predict binary clicks using distributed load balancing.

        Distributes batches across all available providers to avoid rate limits.

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
            # Use mock predictions
            if not self.use_mock:
                _print_fallback("No API clients available; using mock for all chunks.")
            for chunk in _chunked(profiles, self.batch_size):
                clicks.extend(_mock_predict(ad_text, chunk))
            return clicks

        # Distribute chunks across available clients using round-robin
        chunks = list(_chunked(profiles, self.batch_size))
        num_clients = len(self._clients)
        
        print(f"[Distributed] Processing {len(chunks)} batches across {num_clients} providers: {', '.join(self._provider_names)}")
        
        # Create tasks with round-robin client assignment
        async def predict_with_progress(chunk_idx, chunk, pbar):
            client_idx = chunk_idx % num_clients
            client = self._clients[client_idx]
            provider = self._provider_names[client_idx]
            
            result = await client.predict_chunk_async(ad_text, chunk, ad_platform)
            pbar.update(1)
            return (chunk_idx, result, provider)
        
        # Initialize progress bar
        with tqdm(total=len(chunks), desc=f"Distributed processing ({num_clients} providers)", unit="batch") as pbar:
            tasks = [predict_with_progress(i, chunk, pbar) for i, chunk in enumerate(chunks)]
            results = await asyncio.gather(*tasks)
        
        # Sort results by chunk index to maintain order
        results.sort(key=lambda x: x[0])
        
        # Extract clicks and show provider distribution
        provider_counts = {p: 0 for p in self._provider_names}
        for chunk_idx, result, provider in results:
            clicks.extend(result)
            provider_counts[provider] += 1
        
        print(f"[Distributed] Provider usage: {', '.join([f'{p}={c}' for p, c in provider_counts.items()])}")
        
        return clicks


@dataclass
class LLMClickPredictor:
    """Modular batched LLM-based click predictor with support for multiple providers.

    Uses a registry-based client system for easy extensibility. New providers can
    be added by creating a client class that inherits from BaseLLMClient and 
    registering it in CLIENT_REGISTRY.

    NOTE: For distributed load balancing across multiple providers, use DistributedLLMPredictor instead.

    Attributes:
        provider: LLM provider identifier (e.g., "openai", "deepseek", "gemini", "vllm").
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


async def predict_clicks_distributed(ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook", batch_size: int = 50, use_mock: bool = False, providers: Optional[List[str]] = None, models: Optional[Dict[str, str]] = None, api_keys: Optional[Dict[str, str]] = None) -> List[int]:
    """Convenience function for distributed prediction across multiple providers.
    
    This automatically distributes the workload across OpenAI, DeepSeek, and Gemini
    to avoid rate limits and improve throughput.
    
    Args:
        ad_text: Advertisement copy to evaluate.
        profiles: Profiles to score.
        ad_platform: Platform where the ad is shown (facebook, tiktok, amazon).
        batch_size: Number of profiles per request.
        use_mock: If True, use mock predictions.
        providers: List of provider names to use (default: ["openai", "deepseek", "gemini"]).
        models: Dict mapping provider names to model names (optional).
        api_keys: Dict mapping provider names to API keys (optional).
        
    Returns:
        List of 0/1 integers aligned with ``profiles`` order.
    """
    predictor = DistributedLLMPredictor(
        batch_size=batch_size,
        use_mock=use_mock,
        providers=providers or ["openai", "deepseek", "gemini"],
        models=models,
        api_keys=api_keys
    )
    return await predictor.predict_clicks_async(ad_text, profiles, ad_platform)


__all__ = ["LLMClickPredictor", "DistributedLLMPredictor", "predict_clicks_parallel", "predict_clicks_distributed"]

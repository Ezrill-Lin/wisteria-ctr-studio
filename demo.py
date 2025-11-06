import argparse
import asyncio
import csv
import json
import os
import time
from typing import Any, Dict, List

from SiliconSampling.sampler import load_identity_bank, sample_identities
from CTRPrediction.llm_click_model import LLMClickPredictor


def compute_ctr(clicks: List[int]) -> float:
    if not clicks:
        return 0.0
    return sum(1 for x in clicks if x) / float(len(clicks))


def save_results_to_csv(identities: List[Dict[str, Any]], clicks: List[int], output_path: str) -> None:
    """Save prediction results to a CSV file.
    
    Args:
        identities: List of identity profiles.
        clicks: List of corresponding click predictions (0/1).
        output_path: Path to save the CSV file.
    """
    fieldnames = [
        "id",
        "gender",
        "age",
        "region",
        "occupation",
        "annual_salary",
        "liability_status",
        "is_married",
        "health_status",
        "illness",
        "click",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (p, c) in enumerate(zip(identities, clicks)):
            row = {
                "id": i,
                "gender": p.get("gender"),
                "age": p.get("age"),
                "region": p.get("region"),
                "occupation": p.get("occupation"),
                "annual_salary": p.get("annual_salary"),
                "liability_status": p.get("liability_status"),
                "is_married": p.get("is_married"),
                "health_status": p.get("health_status"),
                "illness": p.get("illness", ""),
                "click": c,
            }
            writer.writerow(row)
    print(f"Saved results to {output_path}")


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Silicon sampling CTR demo")
        parser.add_argument("--ad", required=True, help="Textual advertisement content")
        parser.add_argument("--ad-platform", default="facebook", choices=["facebook", "tiktok", "amazon"], help="Platform where the ad is shown (default: facebook)")
        parser.add_argument("--population-size", type=int, default=1000, help="Number of identities to sample")
        parser.add_argument("--identity-bank", default=os.path.join("SiliconSampling", "data", "identity_bank.json"), help="Path to identity bank JSON")
        parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
        parser.add_argument("--provider", choices=["openai", "deepseek", "vllm", "mock"], 
                          default="openai", help="LLM provider to use")
        parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name (for openai, deepseek providers)")
        parser.add_argument("--runpod-model", choices=["llama3.1-8b", "llama3.1-70b"], 
                           default="llama3.1-8b", help="RunPod model selection (llama3.1-8b, llama3.1-70b)")
        parser.add_argument("--batch-size", type=int, default=32, help="Batch size per LLM call")
        parser.add_argument("--use-mock", action="store_true", help="Force mock LLM (no network)")
        parser.add_argument("--use-sync", action="store_true", help="Use synchronous sequential processing instead of async parallel")
        parser.add_argument("--api-key", default=None, help="Explicit API key override for provider")
        parser.add_argument("--out", default=None, help="Optional CSV output of identities and clicks")

        args = parser.parse_args()

    bank = load_identity_bank(args.identity_bank)
    identities = sample_identities(args.population_size, bank, seed=args.seed)

    # Handle vLLM distributed inference via RunPod SDK
    if args.provider == "vllm":
        model = args.runpod_model
        print(f"[vLLM Distributed] Using {model} model (auto-computed pod distribution)")
        
        predictor = LLMClickPredictor(
            provider="vllm",
            model=model,
            batch_size=args.batch_size,
            use_mock=args.use_mock,
            use_async=True,
            api_key=args.api_key
        )
    else:
        # Standard configuration for other providers
        predictor = LLMClickPredictor(
            provider=args.provider,
            model=args.model,
            batch_size=args.batch_size,
            use_mock=args.use_mock,
            use_async=True,
            api_key=args.api_key,
        )

    # Start timing the prediction process
    start_time = time.time()
    
    # Choose between async and sync processing
    if args.use_sync:
        # Use synchronous sequential processing
        clicks = predictor.predict_clicks(args.ad, identities, args.ad_platform)
    else:
        # Use asynchronous parallel processing
        clicks = asyncio.run(predictor.predict_clicks_async(args.ad, identities, args.ad_platform))
    
    end_time = time.time()
    runtime = end_time - start_time
    
    ctr = compute_ctr(clicks)
    
    # Check if fallback to mock was used during prediction
    used_fallback = hasattr(predictor, '_client') and hasattr(predictor._client, 'used_fallback') and predictor._client.used_fallback
    
    model_provider = args.provider if not args.use_mock and not used_fallback else "mock"
    
    # Get the actual model name used
    if args.use_mock or used_fallback:
        model_name = "mock model (no LLM)"
    elif args.provider == "vllm":
        # For vLLM distributed inference, use the actual model that was configured
        model_name = predictor._client.model if hasattr(predictor, '_client') and hasattr(predictor._client, 'model') else args.runpod_model
    else:
        model_name = args.model
    
    print(f"Sampled identities: {len(identities)}")
    print(f"Ad platform: {args.ad_platform}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model Provider: {model_provider} | Model: {model_name}")
    print(f"Processing mode: {'synchronous' if args.use_sync else 'asynchronous parallel'}")
    print(f"Clicks: {sum(clicks)} | Non-clicks: {len(clicks) - sum(clicks)}")
    print(f"CTR: {ctr:.4f}")
    print(f"Runtime: {runtime:.2f} seconds")

    if args.out:
        save_results_to_csv(identities, clicks, args.out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # Hardcoded parameters for quick testing when no CLI args provided
        class Args:
            pass
        args = Args()
        args.ad = '''Discover the ultimate travel experience with our exclusive vacation packages! Book now and save big on your next adventure.'''
        args.ad_platform = "facebook"
        args.population_size = 12000  # Test distributed inference with 12k profiles
        args.batch_size = 32
        args.provider = "vllm"  # Test vLLM distributed inference via RunPod SDK
        args.runpod_model = "llama3.1-8b" 
        args.profiles_per_pod = 5000  # 12k profiles will need 3 pods
        args.use_mock = True  # Use mock for testing without actual API calls
        args.use_sync = False
        args.identity_bank = os.path.join("SiliconSampling", "data", "identity_bank.json")
        args.seed = 42
        args.api_key = None  
        args.out = None
        args.model = "gpt-4o-mini"  # Fallback for other providers
        main(args)
    else:
        main()
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
        parser.add_argument("--provider", choices=["openai", "deepseek", "runpod", "mock"], 
                           default="openai", help="LLM provider (default: openai)")
        parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
        parser.add_argument("--runpod-model", choices=["llama-8b", "llama-70b"], 
                           default="llama-8b", help="RunPod model selection (llama-8b, llama-70b)")
        parser.add_argument("--runpod-endpoint", help="RunPod endpoint ID for serverless")
        parser.add_argument("--auto-model", action="store_true", 
                           help="Auto-select RunPod model based on population size")
        parser.add_argument("--batch-size", type=int, default=50, help="Batch size per LLM call")
        parser.add_argument("--use-mock", action="store_true", help="Force mock LLM (no network)")
        parser.add_argument("--use-sync", action="store_true", help="Use synchronous sequential processing instead of async parallel")
        parser.add_argument("--api-key", default=None, help="Explicit API key override for provider")
        parser.add_argument("--out", default=None, help="Optional CSV output of identities and clicks")

        args = parser.parse_args()

    bank = load_identity_bank(args.identity_bank)
    identities = sample_identities(args.population_size, bank, seed=args.seed)

    # Handle RunPod-specific configuration
    if args.provider == "runpod":
        # Auto-select model based on population size if requested
        if args.auto_model:
            if args.population_size < 5000:
                model = "llama-70b"      # High accuracy for small tests
                print(f"📊 Auto-selected Llama 70B for {args.population_size:,} profiles (high accuracy)")
            else:
                model = "llama-8b"       # Cost-effective for large scale
                print(f"📊 Auto-selected Llama 8B for {args.population_size:,} profiles (cost-effective)")
        else:
            model = args.runpod_model
        
        # Adjust batch size for RunPod (larger batches are more efficient)
        batch_size = max(args.batch_size, 100) if args.population_size > 1000 else args.batch_size
        
        print(f"🏃 Using RunPod with {model} model (batch size: {batch_size})")
        
        predictor = LLMClickPredictor(
            provider=args.provider,
            model=model,
            batch_size=batch_size,
            use_mock=args.use_mock,
            use_async=True,  # Keep this True since we handle sync/async at call level
            api_key=args.api_key,
            runpod_endpoint_id=args.runpod_endpoint
        )
    else:
        # Standard configuration for other providers
        predictor = LLMClickPredictor(
            provider=args.provider,
            model=args.model,
            batch_size=args.batch_size,
            use_mock=args.use_mock,
            use_async=True,  # Keep this True since we handle sync/async at call level
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
    model_provider = args.provider if not args.use_mock else "None"
    model = args.model if not args.use_mock else "mock model (none LLM)"
    
    print(f"Sampled identities: {len(identities)}")
    print(f"Ad platform: {args.ad_platform}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model Provider: {model_provider} | Model: {model}")
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
        args.population_size = 1000
        args.batch_size = 1
        args.provider = "deepseek"  
        args.model = "deepseek-chat" 
        args.use_mock = False
        args.use_sync = False
        args.identity_bank = os.path.join("SiliconSampling", "data", "identity_bank.json")
        args.seed = 42
        args.api_key = None  
        args.out = None  
        main(args)
    else:
        main()
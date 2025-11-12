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


def save_results_to_csv(personas: List[Dict[str, Any]], clicks: List[int], output_path: str) -> None:
    """Save prediction results to a CSV file.
    
    Args:
        personas: List of persona profiles.
        clicks: List of corresponding click predictions (0/1).
        output_path: Path to save the CSV file.
    """
    fieldnames = [
        "id",
        "age",
        "gender",
        "state",
        "race",
        "education",
        "occupation",
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
        "click",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (p, c) in enumerate(zip(personas, clicks)):
            demographics = p.get("demographics", {})
            personality = p.get("personality", {})
            scores = personality.get("scores", {})
            
            row = {
                "id": p.get("id", i),
                "age": demographics.get("age"),
                "gender": demographics.get("gender"),
                "state": demographics.get("state"),
                "race": demographics.get("race"),
                "education": demographics.get("educational_attainment"),
                "occupation": demographics.get("occupation"),
                "openness": scores.get("openness"),
                "conscientiousness": scores.get("conscientiousness"),
                "extraversion": scores.get("extraversion"),
                "agreeableness": scores.get("agreeableness"),
                "neuroticism": scores.get("neuroticism"),
                "click": c,
            }
            writer.writerow(row)
    print(f"Saved results to {output_path}")


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Silicon sampling CTR demo with persona profiles")
        parser.add_argument("--ad", required=True, help="Textual advertisement content")
        parser.add_argument("--ad-platform", default="facebook", choices=["facebook", "tiktok", "amazon"], help="Platform where the ad is shown (default: facebook)")
        parser.add_argument("--population-size", type=int, default=1000, help="Number of personas to sample")
        parser.add_argument("--personas-file", default=os.path.join("SiliconSampling", "generated_personas.json"), help="Path to generated personas JSON")
        parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
        parser.add_argument("--provider", choices=["openai", "deepseek", "vllm"], 
                          default="openai", help="LLM provider to use")
        parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name (for openai, deepseek providers)")
        parser.add_argument("--runpod-model", choices=["llama3.1-8b", "llama3.1-70b"], 
                           default="llama3.1-8b", help="RunPod model selection (llama3.1-8b, llama3.1-70b)")
        parser.add_argument("--batch-size", type=int, default=32, help="Batch size per LLM call")
        parser.add_argument("--use-mock", action="store_true", help="Force mock LLM (no network)")
        parser.add_argument("--api-key", default=None, help="Explicit API key override for provider")
        parser.add_argument("--out", default=None, help="Optional CSV output of personas and clicks")

        args = parser.parse_args()

    # Load generated personas instead of identity bank
    personas = load_identity_bank(args.personas_file)
    print(f"Loaded {len(personas)} personas from {args.personas_file}")
    
    # Sample personas from the loaded list
    sampled_personas = sample_identities(args.population_size, personas, seed=args.seed)

    # Handle vLLM distributed inference via RunPod SDK
    if args.provider == "vllm":
        model = args.runpod_model
        print(f"[vLLM Distributed] Using {model} model (auto-computed pod distribution)")
        
        predictor = LLMClickPredictor(
            provider="vllm",
            model=model,
            batch_size=args.batch_size,
            use_mock=args.use_mock,
            api_key=args.api_key
        )
    else:
        # Standard configuration for other providers
        predictor = LLMClickPredictor(
            provider=args.provider,
            model=args.model,
            batch_size=args.batch_size,
            use_mock=args.use_mock,
            api_key=args.api_key,
        )

    # Start timing the prediction process
    start_time = time.time()
    
    # Use asynchronous parallel processing
    clicks = asyncio.run(predictor.predict_clicks_async(args.ad, sampled_personas, args.ad_platform))
    
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
    
    print(f"Sampled personas: {len(sampled_personas)}")
    print(f"Ad platform: {args.ad_platform}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model Provider: {model_provider} | Model: {model_name}")
    print(f"Clicks: {sum(clicks)} | Non-clicks: {len(clicks) - sum(clicks)}")
    print(f"CTR: {ctr:.4f}")
    print(f"Runtime: {runtime:.2f} seconds")

    if args.out:
        save_results_to_csv(sampled_personas, clicks, args.out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # Hardcoded parameters for quick testing when no CLI args provided
        class Args:
            pass
        args = Args()
        args.ad = '''Discover the ultimate travel experience with our exclusive vacation packages! Book now and save big on your next adventure.'''
        args.ad_platform = "facebook"
        args.population_size = 100  # Test with 100 personas for quick testing
        args.batch_size = 32
        args.provider = "openai"  # Use openai provider but with mock enabled
        args.runpod_model = "llama3.1-8b" 
        args.profiles_per_pod = 5000
        args.use_mock = True  # Force mock mode for testing without API calls
        args.personas_file = os.path.join("SiliconSampling", "generated_personas.json")
        args.seed = 42
        args.api_key = None  
        args.out = None
        args.model = "gpt-4o-mini"  # Fallback for other providers
        main(args)
    else:
        main()
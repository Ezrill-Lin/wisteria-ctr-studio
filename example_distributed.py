"""Example: Using Distributed Load Balancing Mode

This example demonstrates how to use the distributed load balancing feature
to process CTR predictions across multiple LLM providers simultaneously.
"""

import asyncio
import os
from CTRPrediction import DistributedLLMPredictor, predict_clicks_distributed
from SiliconSampling.sampler import load_identity_bank, sample_identities


async def example_basic_distributed():
    """Example 1: Basic distributed prediction"""
    print("="*80)
    print("Example 1: Basic Distributed Prediction")
    print("="*80)
    
    # Load personas
    personas_file = os.path.join("SiliconSampling", "generated_personas.json")
    personas = load_identity_bank(personas_file)
    sampled = sample_identities(100, personas, seed=42)
    
    # Create distributed predictor
    predictor = DistributedLLMPredictor(
        batch_size=20,
        use_mock=True  # Set to False to use real API calls
    )
    
    # Run predictions
    ad_text = "Premium fitness subscription with personal training"
    clicks = await predictor.predict_clicks_async(ad_text, sampled, "facebook")
    
    ctr = sum(clicks) / len(clicks)
    print(f"\nResults: {sum(clicks)} clicks out of {len(clicks)} personas")
    print(f"CTR: {ctr:.2%}\n")


async def example_convenience_function():
    """Example 2: Using the convenience function"""
    print("="*80)
    print("Example 2: Using Convenience Function")
    print("="*80)
    
    # Load personas
    personas_file = os.path.join("SiliconSampling", "generated_personas.json")
    personas = load_identity_bank(personas_file)
    sampled = sample_identities(100, personas, seed=42)
    
    # Use convenience function (one-liner)
    ad_text = "Eco-friendly electric vehicle with tax incentives"
    clicks = await predict_clicks_distributed(
        ad_text=ad_text,
        profiles=sampled,
        ad_platform="facebook",
        batch_size=25,
        use_mock=True,  # Set to False for real API calls
        providers=["openai", "deepseek", "gemini"]
    )
    
    ctr = sum(clicks) / len(clicks)
    print(f"\nResults: {sum(clicks)} clicks out of {len(clicks)} personas")
    print(f"CTR: {ctr:.2%}\n")


async def example_custom_models():
    """Example 3: Custom model configuration"""
    print("="*80)
    print("Example 3: Custom Models Per Provider")
    print("="*80)
    
    # Load personas
    personas_file = os.path.join("SiliconSampling", "generated_personas.json")
    personas = load_identity_bank(personas_file)
    sampled = sample_identities(100, personas, seed=42)
    
    # Create predictor with custom models
    predictor = DistributedLLMPredictor(
        batch_size=20,
        use_mock=True,
        providers=["openai", "deepseek", "gemini"],
        models={
            "openai": "gpt-4o-mini",
            "deepseek": "deepseek-chat",
            "gemini": "gemini-1.5-flash"
        }
    )
    
    ad_text = "AI-powered smart home automation system"
    clicks = await predictor.predict_clicks_async(ad_text, sampled, "amazon")
    
    ctr = sum(clicks) / len(clicks)
    print(f"\nResults: {sum(clicks)} clicks out of {len(clicks)} personas")
    print(f"CTR: {ctr:.2%}\n")


async def example_selective_providers():
    """Example 4: Using only specific providers"""
    print("="*80)
    print("Example 4: Selective Provider Usage")
    print("="*80)
    
    # Load personas
    personas_file = os.path.join("SiliconSampling", "generated_personas.json")
    personas = load_identity_bank(personas_file)
    sampled = sample_identities(100, personas, seed=42)
    
    # Use only DeepSeek and Gemini (cost-effective combination)
    predictor = DistributedLLMPredictor(
        batch_size=20,
        use_mock=True,
        providers=["deepseek", "gemini"]
    )
    
    ad_text = "Budget travel packages to Southeast Asia"
    clicks = await predictor.predict_clicks_async(ad_text, sampled, "tiktok")
    
    ctr = sum(clicks) / len(clicks)
    print(f"\nResults: {sum(clicks)} clicks out of {len(clicks)} personas")
    print(f"CTR: {ctr:.2%}\n")


async def example_large_scale():
    """Example 5: Large-scale prediction (showing benefits)"""
    print("="*80)
    print("Example 5: Large-Scale Prediction")
    print("="*80)
    
    # Load personas
    personas_file = os.path.join("SiliconSampling", "generated_personas.json")
    personas = load_identity_bank(personas_file)
    sampled = sample_identities(1000, personas, seed=42)  # 1000 personas
    
    import time
    
    # Distributed mode
    print("\nUsing Distributed Mode:")
    predictor_dist = DistributedLLMPredictor(
        batch_size=50,
        use_mock=True,
        providers=["openai", "deepseek", "gemini"]
    )
    
    start = time.time()
    ad_text = "Luxury smartwatch with health monitoring"
    clicks_dist = await predictor_dist.predict_clicks_async(ad_text, sampled, "facebook")
    time_dist = time.time() - start
    
    ctr_dist = sum(clicks_dist) / len(clicks_dist)
    print(f"Time: {time_dist:.2f}s | CTR: {ctr_dist:.2%}")
    
    # Note: In mock mode, timing won't show real difference
    # With real API calls, distributed mode would be significantly faster
    print("\nNote: With real API calls, distributed mode processes ~3x faster")
    print("      and avoids rate limits by spreading load across providers!\n")


async def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("DISTRIBUTED LOAD BALANCING EXAMPLES")
    print("="*80 + "\n")
    
    print("Note: These examples use mock mode. To use real API calls:")
    print("  1. Set environment variables: OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY")
    print("  2. Change use_mock=False in the examples\n")
    
    await example_basic_distributed()
    await example_convenience_function()
    await example_custom_models()
    await example_selective_providers()
    await example_large_scale()
    
    print("="*80)
    print("Examples completed!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

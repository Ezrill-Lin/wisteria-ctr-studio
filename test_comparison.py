"""Quick comparison test: Single Provider vs Distributed Mode"""

import asyncio
import time
from CTRPrediction import LLMClickPredictor, DistributedLLMPredictor
from SiliconSampling.sampler import load_identity_bank, sample_identities
import os

async def test_comparison():
    # Load personas
    personas_file = os.path.join("SiliconSampling", "generated_personas.json")
    personas = load_identity_bank(personas_file)
    sampled = sample_identities(100, personas, seed=42)
    
    ad_text = "Premium electric vehicle with advanced autopilot features"
    ad_platform = "facebook"
    
    print("="*80)
    print("COMPARISON TEST: Single Provider vs Distributed Mode")
    print("="*80)
    print(f"Population: {len(sampled)} personas")
    print(f"Advertisement: {ad_text}")
    print(f"Platform: {ad_platform}")
    print("="*80)
    
    # Test 1: Single Provider (OpenAI)
    print("\n[TEST 1] Single Provider Mode (OpenAI only)")
    print("-"*80)
    
    predictor_single = LLMClickPredictor(
        provider="openai",
        model="gpt-4o-mini",
        batch_size=25,
        use_mock=False
    )
    
    start = time.time()
    clicks_single = await predictor_single.predict_clicks_async(ad_text, sampled, ad_platform)
    time_single = time.time() - start
    ctr_single = sum(clicks_single) / len(clicks_single)
    
    print(f"✓ Completed in {time_single:.2f} seconds")
    print(f"  CTR: {ctr_single:.2%} ({sum(clicks_single)} clicks)")
    print(f"  Rate: All {len(sampled)//25} batches sent to OpenAI")
    
    # Test 2: Distributed Mode
    print("\n[TEST 2] Distributed Mode (OpenAI + DeepSeek)")
    print("-"*80)
    
    predictor_dist = DistributedLLMPredictor(
        batch_size=25,
        use_mock=False,
        providers=["openai", "deepseek"]
    )
    
    start = time.time()
    clicks_dist = await predictor_dist.predict_clicks_async(ad_text, sampled, ad_platform)
    time_dist = time.time() - start
    ctr_dist = sum(clicks_dist) / len(clicks_dist)
    
    print(f"✓ Completed in {time_dist:.2f} seconds")
    print(f"  CTR: {ctr_dist:.2%} ({sum(clicks_dist)} clicks)")
    print(f"  Rate: ~{(len(sampled)//25)//2} batches per provider")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Single Provider Time:  {time_single:.2f}s")
    print(f"Distributed Time:      {time_dist:.2f}s")
    
    if time_dist < time_single:
        speedup = (time_single / time_dist - 1) * 100
        print(f"Speedup:               {speedup:.1f}% faster ⚡")
    
    print(f"\nRate Limit Benefit:")
    print(f"  Single:      {len(sampled)//25} batches → 1 provider")
    print(f"  Distributed: ~{(len(sampled)//25)//2} batches → each provider")
    print(f"  Risk Reduction: ~50% per provider ✓")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_comparison())

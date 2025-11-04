"""Quick test to see if fallback messages are showing."""

import os
from CTRPrediction import LLMClickPredictor

# Test with RunPod - should hit deleted endpoint and fallback
predictor = LLMClickPredictor(
    provider="runpod",
    model="llama-8b",
    batch_size=5,
    use_async=False  # Use sync for simpler testing
)

# Create minimal test data
test_ad = "Buy our product now!"
test_profiles = [
    {"gender": "M", "age": 25, "region": "NYC", "occupation": "Engineer", 
     "annual_salary": 80000, "liability_status": 5000, "is_married": False,
     "health_status": True, "illness": None}
]

print("=" * 80)
print("Testing RunPod client with deleted endpoint...")
print("=" * 80)

# This should attempt to connect and fallback
results = predictor.predict_clicks(test_ad, test_profiles)

print("\n" + "=" * 80)
print(f"Results: {results}")
print("=" * 80)

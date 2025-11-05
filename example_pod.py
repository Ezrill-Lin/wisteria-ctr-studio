"""
Example: Using RunPod Pod directly with custom URL
"""

from CTRPrediction import LLMClickPredictor
from SiliconSampling import sample_identities, load_identity_bank
import os
import requests

# Your Pod's vLLM endpoint URL
POD_URL = os.getenv("RUNPOD_LLAMA_8B_URL")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

print(f"Pod URL: {POD_URL}")
print(f"API Key: {'Set' if RUNPOD_API_KEY else 'Not set'}")

# First, let's check what models are available on the Pod
if POD_URL:
    try:
        models_url = f"{POD_URL}/models"
        headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"} if RUNPOD_API_KEY else {}
        response = requests.get(models_url, headers=headers, timeout=10)
        print(f"Models endpoint response: {response.status_code}")
        if response.status_code == 200:
            models_data = response.json()
            print("Available models:")
            for model in models_data.get("data", []):
                print(f"  - {model.get('id', 'unknown')}")
        else:
            print(f"Models endpoint error: {response.text}")
    except Exception as e:
        print(f"Error checking models: {e}")

# Create predictor with Pod URL
predictor = LLMClickPredictor(
    provider="runpod",
    model="llama-8b",
    batch_size=10,
    use_async=True,
    api_key=RUNPOD_API_KEY,
)

# Override the HTTP base URL to use your Pod
predictor._client.http_base_url = POD_URL
predictor._client.mode = "http"

# Override the model name for vLLM endpoints (they typically use simpler names)
# Use the exact model name from the /models endpoint
predictor._client.model_configs["llama-8b"]["model_name"] = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Sample some identities
identity_bank_path = "SiliconSampling/data/identity_bank.json"
identity_bank = load_identity_bank(identity_bank_path)
profiles = sample_identities(5, identity_bank, seed=42)  # Use smaller sample for testing

print(f"\nTesting with {len(profiles)} profiles...")

# Run prediction
ad_text = "Get 50% off on all winter jackets! Limited time offer."
clicks = predictor.predict_clicks(ad_text, profiles)

print(f"Clicks: {sum(clicks)}")
print(f"CTR: {sum(clicks) / len(clicks):.2%}")

# Let's also test a single prediction to see what the raw response looks like
print("\n--- Testing batch prediction with proper prompt ---")
if POD_URL:
    try:
        # Let's see what prompt is being generated
        from CTRPrediction.base_client import _build_prompt
        test_profiles = [
            {"gender": "female", "age": 25, "region": "New York, NY", "occupation": "engineer", "annual_salary": 75000, "liability_status": 5000, "is_married": False, "health_status": False},
            {"gender": "male", "age": 35, "region": "Los Angeles, CA", "occupation": "teacher", "annual_salary": 55000, "liability_status": 2000, "is_married": True, "health_status": False}
        ]
        
        prompt = _build_prompt("Get 50% off on all winter jackets!", test_profiles)
        print("Generated prompt:")
        print("=" * 50)
        print(prompt)
        print("=" * 50)
        
        # Make a direct API call with the proper prompt
        test_payload = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 50,  # Increase max_tokens to allow for JSON array response
            "temperature": 0.1
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RUNPOD_API_KEY}"
        }
        
        response = requests.post(f"{POD_URL}/chat/completions", 
                               json=test_payload, 
                               headers=headers, 
                               timeout=30)
        
        print(f"Batch API response status: {response.status_code}")
        if response.status_code == 200:
            resp_data = response.json()
            print("Raw batch API response content:")
            print(repr(resp_data["choices"][0]["message"]["content"]))
        else:
            print(f"API error: {response.text}")
            
    except Exception as e:
        print(f"Batch API test error: {e}")


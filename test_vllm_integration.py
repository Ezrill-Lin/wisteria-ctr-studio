#!/usr/bin/env python3
"""Test script for vLLM integration with Llama models.

This script tests the vLLM client implementation with mock and real endpoints.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from CTRPrediction.vllm_client import VLLMClient
from CTRPrediction.llm_click_model import LLMClickPredictor


def test_vllm_client_basic():
    """Test basic vLLM client functionality."""
    print("🧪 Testing vLLM Client Basic Functionality")
    print("=" * 50)
    
    # Test client initialization
    for model in ["llama-8b", "llama-70b"]:
        print(f"\n📋 Testing {model} client initialization...")
        try:
            client = VLLMClient(
                base_url="http://localhost:8000",  # Mock endpoint
                model=model
            )
            
            info = client.get_model_info()
            print(f"✅ Client initialized successfully:")
            print(f"   Model: {info['model']}")
            print(f"   Full name: {info['full_model_name']}")
            print(f"   Description: {info['description']}")
            
        except Exception as e:
            print(f"❌ Client initialization failed: {e}")


def test_llm_predictor_integration():
    """Test LLMClickPredictor integration with vLLM."""
    print("\n🔗 Testing LLMClickPredictor Integration")
    print("=" * 50)
    
    # Sample test profiles
    test_profiles = [
        {
            "gender": "Female",
            "age": 28,
            "region": "California",
            "occupation": "Software Engineer",
            "annual_salary": 120000,
            "is_married": False,
            "health_status": True
        },
        {
            "gender": "Male", 
            "age": 45,
            "region": "Texas",
            "occupation": "Teacher", 
            "annual_salary": 50000,
            "is_married": True,
            "health_status": True
        }
    ]
    
    ad_text = "Revolutionary fitness tracker - monitor your health 24/7!"
    
    for model in ["llama-8b", "llama-70b"]:
        print(f"\n🎯 Testing {model} predictor...")
        try:
            predictor = LLMClickPredictor(
                provider="vllm",
                model=model,
                batch_size=10,
                use_mock=True,  # Use mock for testing
                vllm_base_url="http://localhost:8000"
            )
            
            # Test synchronous prediction
            print(f"   🔄 Running sync prediction...")
            clicks = predictor.predict_clicks(ad_text, test_profiles)
            print(f"   ✅ Sync prediction: {clicks}")
            
            # Test asynchronous prediction
            print(f"   🔄 Running async prediction...")
            clicks_async = asyncio.run(
                predictor.predict_clicks_async(ad_text, test_profiles)
            )
            print(f"   ✅ Async prediction: {clicks_async}")
            
        except Exception as e:
            print(f"   ❌ Predictor test failed: {e}")


def test_demo_integration():
    """Test demo.py integration with vLLM arguments."""
    print("\n🎬 Testing Demo Integration")
    print("=" * 50)
    
    # Import demo main function
    try:
        from demo import main
        
        # Test different vLLM configurations
        test_configs = [
            # Basic vLLM usage
            ["--ad", "Great coffee deal!", "--provider", "vllm", "--population-size", "10"],
            
            # Specific model selection
            ["--ad", "Health insurance!", "--provider", "vllm", "--vllm-model", "llama-70b", "--population-size", "5"],
            
            # Auto model selection
            ["--ad", "Tech gadget!", "--provider", "vllm", "--auto-model", "--population-size", "1000"],
            
            # Custom vLLM URL
            ["--ad", "Fashion sale!", "--provider", "vllm", "--vllm-url", "http://custom-vllm:8000", "--population-size", "3"]
        ]
        
        for i, config in enumerate(test_configs, 1):
            print(f"\n📝 Test {i}: {' '.join(config)}")
            try:
                # Add mock flag for testing
                config.extend(["--use-mock"])
                main(config)
                print(f"   ✅ Demo test {i} passed")
                
            except Exception as e:
                print(f"   ❌ Demo test {i} failed: {e}")
    
    except ImportError as e:
        print(f"❌ Could not import demo module: {e}")


def test_environment_variables():
    """Test environment variable handling."""
    print("\n🌍 Testing Environment Variables")
    print("=" * 50)
    
    # Test VLLM_BASE_URL
    original_url = os.getenv("VLLM_BASE_URL")
    
    try:
        # Set test environment
        os.environ["VLLM_BASE_URL"] = "http://test-vllm:9000"
        
        client = VLLMClient(model="llama-8b")
        info = client.get_model_info()
        
        print(f"✅ Environment URL test:")
        print(f"   Expected: http://test-vllm:9000")
        print(f"   Actual: {info['base_url']}")
        
        # Test override
        client_override = VLLMClient(
            base_url="http://override-vllm:8000",
            model="llama-8b"
        )
        info_override = client_override.get_model_info()
        
        print(f"✅ URL override test:")
        print(f"   Expected: http://override-vllm:8000") 
        print(f"   Actual: {info_override['base_url']}")
        
    finally:
        # Restore original environment
        if original_url is not None:
            os.environ["VLLM_BASE_URL"] = original_url
        elif "VLLM_BASE_URL" in os.environ:
            del os.environ["VLLM_BASE_URL"]


def main():
    """Run all tests."""
    print("🚀 vLLM Integration Test Suite")
    print("=" * 60)
    print("This script tests the vLLM integration without requiring")
    print("actual vLLM servers to be running (uses mock predictions).")
    print("=" * 60)
    
    test_vllm_client_basic()
    test_llm_predictor_integration()
    test_demo_integration() 
    test_environment_variables()
    
    print("\n🎉 Test Suite Complete!")
    print("\n📝 Next Steps:")
    print("1. Deploy vLLM servers: cd deploy/vllm && ./setup-cluster.sh")
    print("2. Apply Kubernetes configs: kubectl apply -f .")
    print("3. Get external IP: kubectl get ingress vllm-ingress -n vllm")
    print("4. Test with real vLLM: python demo.py --provider vllm --vllm-url http://EXTERNAL_IP")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test script for RunPod vLLM integration
Validates RunPod setup and endpoint functionality
"""

import os
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional

# Add project root to path
import sys
import pathlib
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from CTRPrediction.runpod_client import RunPodClient
from CTRPrediction.llm_click_model import LLMClickPredictor


@dataclass
class TestResults:
    """Test results container"""
    test_name: str
    success: bool
    response_time: float
    error: Optional[str] = None
    response: Optional[str] = None


class RunPodTester:
    """Comprehensive RunPod testing suite"""
    
    def __init__(self):
        self.results = []
        
    def log_result(self, result: TestResults):
        """Log test result"""
        self.results.append(result)
        status = "✅ PASS" if result.success else "❌ FAIL"
        print(f"{status} {result.test_name} ({result.response_time:.2f}s)")
        
        if result.error:
            print(f"   Error: {result.error}")
        elif result.response:
            preview = result.response[:100] + "..." if len(result.response) > 100 else result.response
            print(f"   Response: {preview}")
        print()
    
    def test_environment(self) -> TestResults:
        """Test environment variables"""
        start_time = time.time()
        
        api_key = os.getenv('RUNPOD_API_KEY')
        endpoint_8b = os.getenv('RUNPOD_LLAMA_8B_ENDPOINT')
        endpoint_70b = os.getenv('RUNPOD_LLAMA_70B_ENDPOINT')
        
        if not api_key:
            return TestResults(
                "Environment - API Key",
                False,
                time.time() - start_time,
                "RUNPOD_API_KEY not found"
            )
        
        if not endpoint_8b and not endpoint_70b:
            return TestResults(
                "Environment - Endpoints",
                False,
                time.time() - start_time,
                "No endpoint IDs found (RUNPOD_LLAMA_8B_ENDPOINT or RUNPOD_LLAMA_70B_ENDPOINT)"
            )
        
        return TestResults(
            "Environment - Configuration",
            True,
            time.time() - start_time,
            response=f"API Key: {api_key[:8]}..., 8B Endpoint: {endpoint_8b}, 70B Endpoint: {endpoint_70b}"
        )
    
    async def test_runpod_client_init(self) -> TestResults:
        """Test RunPod client initialization"""
        start_time = time.time()
        
        try:
            client = RunPodClient()
            return TestResults(
                "RunPod Client - Initialization",
                True,
                time.time() - start_time,
                response="Client initialized successfully"
            )
        except Exception as e:
            return TestResults(
                "RunPod Client - Initialization",
                False,
                time.time() - start_time,
                str(e)
            )
    
    async def test_llama_8b_prediction(self) -> TestResults:
        """Test Llama 8B prediction"""
        start_time = time.time()
        
        if not os.getenv('RUNPOD_LLAMA_8B_ENDPOINT'):
            return TestResults(
                "Llama 8B - Prediction",
                False,
                time.time() - start_time,
                "RUNPOD_LLAMA_8B_ENDPOINT not configured"
            )
        
        try:
            predictor = LLMClickPredictor(
                provider='runpod',
                runpod_model='llama-8b',
                runpod_pod_type='serverless'
            )
            
            # Simple test prediction
            test_ad = "Premium coffee beans - freshly roasted daily"
            test_user = {"age": 25, "gender": "female", "interests": ["coffee", "food"]}
            
            prediction = await predictor.predict_click_async(test_ad, test_user)
            
            return TestResults(
                "Llama 8B - Prediction",
                True,
                time.time() - start_time,
                response=f"Prediction: {prediction}"
            )
        except Exception as e:
            return TestResults(
                "Llama 8B - Prediction", 
                False,
                time.time() - start_time,
                str(e)
            )
    
    async def test_llama_70b_prediction(self) -> TestResults:
        """Test Llama 70B prediction"""
        start_time = time.time()
        
        if not os.getenv('RUNPOD_LLAMA_70B_ENDPOINT'):
            return TestResults(
                "Llama 70B - Prediction",
                False,
                time.time() - start_time,
                "RUNPOD_LLAMA_70B_ENDPOINT not configured"
            )
        
        try:
            predictor = LLMClickPredictor(
                provider='runpod',
                runpod_model='llama-70b',
                runpod_pod_type='serverless'
            )
            
            # Simple test prediction
            test_ad = "Luxury watch collection - Swiss made timepieces"
            test_user = {"age": 45, "gender": "male", "interests": ["luxury", "watches"]}
            
            prediction = await predictor.predict_click_async(test_ad, test_user)
            
            return TestResults(
                "Llama 70B - Prediction",
                True,
                time.time() - start_time,
                response=f"Prediction: {prediction}"
            )
        except Exception as e:
            return TestResults(
                "Llama 70B - Prediction",
                False,
                time.time() - start_time,
                str(e)
            )
    
    async def test_batch_predictions(self) -> TestResults:
        """Test batch predictions"""
        start_time = time.time()
        
        try:
            # Use 8B model for batch test (faster)
            predictor = LLMClickPredictor(
                provider='runpod',
                runpod_model='llama-8b',
                runpod_pod_type='serverless'
            )
            
            test_cases = [
                ("Gaming laptop with RGB keyboard", {"age": 22, "gender": "male", "interests": ["gaming", "tech"]}),
                ("Organic skincare products", {"age": 30, "gender": "female", "interests": ["beauty", "health"]}),
                ("Professional cooking knives", {"age": 35, "gender": "male", "interests": ["cooking", "kitchen"]}),
            ]
            
            predictions = []
            for ad, user in test_cases:
                prediction = await predictor.predict_click_async(ad, user)
                predictions.append(prediction)
            
            return TestResults(
                "Batch Predictions - 3 samples",
                True,
                time.time() - start_time,
                response=f"Predictions: {predictions}"
            )
        except Exception as e:
            return TestResults(
                "Batch Predictions - 3 samples",
                False,
                time.time() - start_time,
                str(e)
            )
    
    async def test_error_handling(self) -> TestResults:
        """Test error handling with invalid inputs"""
        start_time = time.time()
        
        try:
            # Test with invalid model
            try:
                predictor = LLMClickPredictor(
                    provider='runpod',
                    runpod_model='invalid-model',
                    runpod_pod_type='serverless'
                )
                await predictor.predict_click_async("Test ad", {"age": 25})
                
                return TestResults(
                    "Error Handling - Invalid Model",
                    False,
                    time.time() - start_time,
                    "Should have failed with invalid model"
                )
            except Exception:
                # Expected to fail
                return TestResults(
                    "Error Handling - Invalid Model",
                    True,
                    time.time() - start_time,
                    response="Correctly handled invalid model"
                )
        except Exception as e:
            return TestResults(
                "Error Handling - Invalid Model",
                False,
                time.time() - start_time,
                str(e)
            )
    
    def print_summary(self):
        """Print test summary"""
        passed = sum(1 for r in self.results if r.success)
        total = len(self.results)
        
        print("=" * 60)
        print(f"🧪 RunPod Test Results: {passed}/{total} tests passed")
        print("=" * 60)
        
        if passed == total:
            print("🎉 All tests passed! RunPod integration is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the errors above.")
            
        print(f"\nTotal test time: {sum(r.response_time for r in self.results):.2f}s")
        
        # Cost estimation
        total_predictions = sum(1 for r in self.results if "Prediction" in r.test_name and r.success)
        if total_predictions > 0:
            cost_8b = total_predictions * 0.0001  # Rough estimate
            print(f"Estimated cost for {total_predictions} predictions: ~${cost_8b:.4f}")


async def main():
    """Run all tests"""
    print("🧪 RunPod vLLM Integration Test Suite")
    print("=" * 60)
    print()
    
    tester = RunPodTester()
    
    # Run tests sequentially
    tests = [
        tester.test_environment,
        tester.test_runpod_client_init,
        tester.test_llama_8b_prediction,
        tester.test_llama_70b_prediction,
        tester.test_batch_predictions,
        tester.test_error_handling,
    ]
    
    for test in tests:
        if asyncio.iscoroutinefunction(test):
            result = await test()
        else:
            result = test()
        tester.log_result(result)
    
    tester.print_summary()
    
    print("\n📋 Next steps:")
    print("1. If tests failed, check your RunPod API key and endpoint configuration")
    print("2. Run production tests: python demo.py --provider runpod --runpod-model llama-8b")
    print("3. Monitor costs at: https://runpod.io/console/billing")


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""Example: Using RunPod SDK for automatic pod creation.

This example demonstrates how to use the enhanced RunPod client
to automatically create and manage pods for CTR prediction.
"""

import os
from CTRPrediction.llm_click_model import LLMClickPredictor
from SiliconSampling.sampler import sample_identities

def main():
    """Example of using RunPod SDK features."""
    
    # Sample ad and population
    ad_text = "Experience the future of mobile technology with our latest smartphone!"
    population_size = 100
    
    print("=== RunPod SDK Auto-Creation Example ===\n")
    print(f"Ad: {ad_text}")
    print(f"Population size: {population_size}")
    print()
    
    # Check if API key is available
    if not os.getenv("RUNPOD_API_KEY"):
        print("❌ No RUNPOD_API_KEY found - using existing pod or mock mode")
        print("💡 Set RUNPOD_API_KEY to enable automatic pod creation")
        auto_create = False
    else:
        print("✅ RUNPOD_API_KEY found - auto-creation available")
        auto_create = True
    
    # Create predictor with auto-creation enabled
    print("\n1. Creating LLM predictor with RunPod SDK support...")
    predictor = LLMClickPredictor(
        provider="vllm",  # Use vLLM provider alias
        model="llama3.1-8b", 
        batch_size=50,
        auto_create_pod=auto_create,  # Enable auto-creation if API key available
        use_async=True
    )
    
    # Get model information
    client_info = predictor._client.get_model_info()
    print(f"   Mode: {client_info['mode']}")
    print(f"   Model: {client_info['model_name']}")
    print(f"   Auto-create: {client_info['auto_create_pod']}")
    if client_info.get('created_pod_id'):
        print(f"   Created pod: {client_info['created_pod_id']}")
    
    # Sample identities
    print("\n2. Sampling identities...")
    import json
    
    # Load identity bank
    identity_bank_path = os.path.join("SiliconSampling", "data", "identity_bank.json")
    if os.path.exists(identity_bank_path):
        with open(identity_bank_path, 'r') as f:
            identity_bank = json.load(f)
        identities = sample_identities(population_size, identity_bank, seed=42)
        print(f"   Sampled {len(identities)} identities")
    else:
        print("   ❌ Identity bank not found - using mock identities")
        identities = [{"age": 25, "gender": "M", "interests": ["tech"]} for _ in range(population_size)]
    
    # Make predictions
    print("\n3. Making CTR predictions...")
    
    # Use context manager for automatic cleanup
    with predictor._client as client:
        if client.mode == "sdk" and auto_create:
            print("   🚀 Auto-creating RunPod pod...")
        
        # Predict clicks
        clicks = predictor.predict_clicks(ad_text, identities, "facebook")
        
        # Calculate results
        total_clicks = sum(clicks)
        ctr = total_clicks / len(clicks) if clicks else 0
        
        print(f"   Total clicks: {total_clicks}")
        print(f"   CTR: {ctr:.4f}")
        
        # Show pod information if created
        if hasattr(client, 'created_pod_id') and client.created_pod_id:
            print(f"   Pod ID: {client.created_pod_id}")
            print(f"   Endpoint: {client.pod_endpoint}")
    
    print("\n✅ Example completed!")
    print("\n💡 Tips:")
    print("   • Pods are automatically cleaned up when using context managers")
    print("   • Set auto_create_pod=False to use existing pods only")
    print("   • Use manual pod management for fine-grained control")

def example_manual_pod_management():
    """Example of manual pod management."""
    print("\n=== Manual Pod Management Example ===\n")
    
    if not os.getenv("RUNPOD_API_KEY"):
        print("❌ Skipping manual pod management - no API key")
        return
    
    from CTRPrediction.runpod_client import RunPodClient
    
    # Create client without auto-creation
    client = RunPodClient(model="llama3.1-8b", auto_create_pod=False)
    print(f"Client mode: {client.mode}")
    
    # List existing pods
    pods = client.list_pods()
    print(f"Existing pods: {len(pods)}")
    
    # Custom pod configuration
    custom_config = {
        "container_disk_in_gb": 30,
        "volume_in_gb": 60,
    }
    
    print("\nTo manually create a pod:")
    print("  success = client.create_pod(custom_config)")
    print("  if success:")
    print("      # Use the pod")
    print("      client.terminate_pod()  # Clean up when done")

if __name__ == "__main__":
    main()
    example_manual_pod_management()
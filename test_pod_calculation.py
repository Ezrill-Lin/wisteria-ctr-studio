#!/usr/bin/env python3
"""
Test script to demonstrate pod calculation for population size of 12,000
"""

def calculate_required_pods(population_size, profiles_per_pod):
    """Calculate required pods using the same formula as RunPodSDKConfig"""
    return max(1, (population_size + profiles_per_pod - 1) // profiles_per_pod)

def calculate_used_pods(active_pod_endpoints, population_size, profiles_per_pod):
    """Calculate used pods in distributed inference"""
    required = calculate_required_pods(population_size, profiles_per_pod)
    return min(len(active_pod_endpoints), required)

def test_pod_calculation():
    """Test pod calculations for different scenarios"""
    population_size = 12000
    profiles_per_pod = 5000
    
    print(f"Testing with population size: {population_size}")
    print(f"Profiles per pod setting: {profiles_per_pod}")
    print("-" * 50)
    
    # Calculate required pods
    required_pods = calculate_required_pods(population_size, profiles_per_pod)
    print(f"Required pods: {required_pods}")
    print(f"Calculation: max(1, ({population_size} + {profiles_per_pod} - 1) // {profiles_per_pod})")
    print(f"            = max(1, {population_size + profiles_per_pod - 1} // {profiles_per_pod})")
    print(f"            = max(1, {(population_size + profiles_per_pod - 1) // profiles_per_pod})")
    print(f"            = {required_pods}")
    print()
    
    # Test different scenarios of available pods
    scenarios = [
        ("Insufficient pods", 1),
        ("Exactly enough pods", required_pods),
        ("More than enough pods", required_pods + 2),
    ]
    
    for scenario_name, available_pods in scenarios:
        # Simulate active pod endpoints
        active_pod_endpoints = [f"https://pod-{i}.example.com" for i in range(available_pods)]
        
        used_pods = calculate_used_pods(active_pod_endpoints, population_size, profiles_per_pod)
        profiles_per_actual_pod = population_size // used_pods if used_pods > 0 else 0
        remainder = population_size % used_pods if used_pods > 0 else 0
        
        print(f"{scenario_name}:")
        print(f"  Available pods: {available_pods}")
        print(f"  Used pods: {used_pods}")
        print(f"  Profiles per pod: ~{profiles_per_actual_pod} ({remainder} remainder)")
        
        # Show how profiles would be distributed
        if used_pods > 0:
            import numpy as np
            profiles = list(range(population_size))  # Mock profiles
            chunks = np.array_split(profiles, used_pods)
            
            print(f"  Chunk distribution:")
            for i, chunk in enumerate(chunks):
                print(f"    Pod {i+1}: {len(chunk)} profiles")
        
        print()

if __name__ == "__main__":
    test_pod_calculation()
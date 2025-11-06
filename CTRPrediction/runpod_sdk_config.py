import asyncio
import time
import runpod
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI
from .runpod_client import RunPodClient

# Disable OpenAI HTTP request logging
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class RunPodSDKConfig(RunPodClient):
    def __init__(self, api_key=None, model="llama3.1-70b", profiles_per_pod=5000, population_size=None):
        super().__init__(model)
        
        # Set API key from parameter or environment variable
        if api_key:
            runpod.api_key = api_key
        elif not runpod.api_key:
            # Try to load from environment if not already set
            import os
            env_key = os.getenv('RUNPOD_API_KEY')
            if env_key:
                runpod.api_key = env_key
        
        self.profiles_per_pod = profiles_per_pod
        self.active_pod_endpoints = []
        self.created_pod_ids = []
        self.total_population_size = population_size  # Track expected total population
        # Check if we can use the RunPod SDK
        self.mode = "sdk" if runpod.api_key else "mock"
        
        print(f"[RunPod SDK] Mode: {self.mode} | API Key: {'SET' if runpod.api_key else 'NOT SET'}")
        if self.total_population_size:
            print(f"[RunPod SDK] Population size: {self.total_population_size}")

    def set_total_population_size(self, size):
        """Set the expected total population size for optimal pod calculation."""
        self.total_population_size = size

    def predict_chunk(self, ad_text, profiles, ad_platform="facebook"):
        """Synchronous wrapper for distributed inference."""
        return asyncio.run(self.predict_chunk_async(ad_text, profiles, ad_platform))

    async def _setup_optimal_pods(self, population_size):
        """Combined method to discover, calculate, and provision optimal pods.
        
        Args:
            population_size: Total number of profiles to process
            
        Returns:
            True if sufficient pods are ready, False otherwise
        """
        # First discover existing running pods
        existing_pods = await self._discover_existing_pods()
        available_pods_count = len(existing_pods)
        
        # Calculate optimal distribution
        m = population_size
        n = available_pods_count
        
        # Only edge case: no running pods (avoid division by zero)
        if n == 0:
            # No existing pods, create minimum needed
            min_needed = max(1, (m + 4096 - 1) // 4096)
            print(f"[Pod Setup] No existing pods, creating {min_needed} new pods")
            return await self._create_pods_only(min_needed)
        
        # Apply the core algorithm to determine if existing pods are sufficient
        per_pod = m / n
        
        if per_pod < 4096:
            # Not overloaded - existing pods are sufficient, just use optimal subset
            optimal_pods = 1
            for i in range(n):
                if m / (n - i) > 128:
                    optimal_pods = n - i 
                    break
            
            print(f"[Pod Setup] Using {optimal_pods} of {n} existing pods (per_pod: {m/optimal_pods:.1f})")
            
            # Add existing running pods to active endpoints (up to optimal count)
            for pod_id, url in existing_pods[:optimal_pods]:
                if url not in self.active_pod_endpoints:
                    self.active_pod_endpoints.append(url)
            
            return len(self.active_pod_endpoints) >= optimal_pods
        else:
            # Overloaded - need to create additional pods
            i = 0
            while m / (n + i) > 4096:
                i += 1
            total_needed = n + i
            pods_to_create = i
            
            print(f"[Pod Setup] Existing pods overloaded (per_pod: {per_pod:.1f}), creating {pods_to_create} additional pods")
            
            # Add all existing running pods to active endpoints first
            for pod_id, url in existing_pods:
                if url not in self.active_pod_endpoints:
                    self.active_pod_endpoints.append(url)
            
            # Then create additional pods
            return await self._create_pods_only(pods_to_create)

    async def _create_pods_only(self, pods_to_create):
        """Create the specified number of pods and add them to active endpoints."""
        if pods_to_create <= 0:
            return True
            
        for i in range(pods_to_create):
            url = await self._create_and_wait_for_pod(len(self.active_pod_endpoints))
            if url:
                self.active_pod_endpoints.append(url)
            else:
                print(f"[Pod Creation] Failed to create pod {i+1}/{pods_to_create}")
                return False
        
        return True

    async def predict_chunk_async(self, ad_text, profiles, ad_platform="facebook"):
        if not profiles:
            return []
        if self.mode == "mock" or not runpod.api_key:
            return self._fallback_to_mock(ad_text, profiles, "SDK not available")
        
        # Set up pods if not already done
        if not self.active_pod_endpoints:
            # Use the provided total population size for optimal pod calculation
            print(f"[Pod Setup] Using population size: {self.total_population_size}")
            if not await self._setup_optimal_pods(self.total_population_size):
                return self._fallback_to_mock(ad_text, profiles, "Failed to provision pods")
        
        return await self._distributed_inference(profiles, ad_text, ad_platform)

    async def _discover_existing_pods(self):
        """Find existing running pods that can be used for CTR inference."""
        if not runpod.api_key:
            return []
        
        try:
            all_pods = runpod.get_pods()
            running_pods = []
            
            for pod in all_pods:
                pod_id = pod["id"]
                pod_status = pod.get("desiredStatus")
                
                if pod_status == "RUNNING":
                    # Check if pod has HTTP ports available
                    ports = pod.get("runtime", {}).get("ports", [])
                    for port in ports:
                        if port.get("type") == "http":
                            # Use port 8000 for vLLM services
                            http_url = f"https://{pod_id}-8000.proxy.runpod.net"
                            
                            # Assume pod is ready if it has RUNNING status and HTTP port
                            running_pods.append((pod_id, http_url))
                            break
            
            return running_pods
            
        except Exception as e:
            print(f"[ERROR] Failed to discover existing pods: {e}")
            return []

    async def _create_and_wait_for_pod(self, pod_index):
        """Create and wait for pod to be ready."""
        if not runpod.api_key:
            return None
        
        config = self.model_configs[self.model]
        pod_name = f"ctr-studio-{self.model}-{pod_index}-{int(time.time())}"
        
        try:
            pod = runpod.create_pod(
                name=pod_name,
                image_name=config["pod_template"],
                gpu_type_id=config["gpu_type"],
                gpu_count=config["gpu_count"],
                container_disk_in_gb=20,
                volume_in_gb=50,
                volume_mount_path="/workspace",
                docker_args=config["docker_args"],
                ports="8000/http",
                env={
                    "HF_TOKEN": "{{ RUNPOD_SECRET_hf_token }}",
                    "HF_HOME": "/workspace/hf_home"
                }
            )
            
            if "id" not in pod:
                raise Exception(f"Pod creation failed: {pod}")
            
            pod_id = pod["id"]
            self.created_pod_ids.append(pod_id)
            return await self._wait_for_pod_ready(pod_id, pod_index)
            
        except Exception as e:
            print(f"[ERROR] Failed to create pod {pod_index+1}: {e}")
            return None

    async def _wait_for_pod_ready(self, pod_id, pod_index):
        """Wait for pod to be ready and return HTTP URL."""
        max_wait = 900  # 30 minutes
        check_interval = 30  # 30 seconds
        
        # Wait for pod to be ready (reduced logging)
        
        for attempt in range(max_wait // check_interval):
            try:
                pod_status = runpod.get_pod(pod_id)
                
                # Handle case where get_pod returns None (pod not yet visible)
                if pod_status is None:
                    # Pod not visible yet, continue waiting
                    await asyncio.sleep(check_interval)
                    continue
                
                # Check if pod is running
                if pod_status.get("desiredStatus") == "RUNNING":
                    runtime = pod_status.get("runtime", {})
                    ports = runtime.get("ports", [])
                    
                    for port in ports:
                        if port.get("type") == "http":
                            # For vLLM services, always use port 8000
                            url = f"https://{pod_id}-8000.proxy.runpod.net"
                            # Assume pod is ready if it has RUNNING status and HTTP port
                            return url
                
                print(f"[Pod {pod_index+1}] Status: {pod_status.get('desiredStatus', 'unknown')}, waiting...")
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                print(f"[ERROR] Pod {pod_index+1} status check failed: {e}")
                await asyncio.sleep(check_interval)
        
        print(f"[Pod {pod_index+1}] Timeout waiting for pod to be ready")
        return None

    async def _distributed_inference(self, profiles, ad_text, ad_platform):
        """Execute distributed inference across pods with even distribution."""
        num_pods = len(self.active_pod_endpoints)
        
        # Calculate even distribution and split into chunks
        chunks = np.array_split(profiles, num_pods)
        chunks = [chunk.tolist() if hasattr(chunk, 'tolist') else list(chunk) for chunk in chunks]
        
        # Process all chunks concurrently
        tasks = [
            self._process_pod_chunk(pod_url, chunk, ad_text, ad_platform, i)
            for i, (pod_url, chunk) in enumerate(zip(self.active_pod_endpoints, chunks))
            if len(chunk) > 0
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        all_predictions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                all_predictions.extend([0] * len(chunks[i]))
            elif isinstance(result, list):
                all_predictions.extend(result)
            else:
                all_predictions.extend([0] * len(chunks[i]))
        
        return all_predictions

    async def _process_pod_chunk(self, pod_url, profiles, ad_text, ad_platform, pod_index):
        """Process a chunk on a single pod using OpenAI client."""
        if not profiles:
            return []
        
        prompt = self._build_prompt(ad_text, profiles, ad_platform)
        
        try:
            # Create OpenAI client for this pod
            client = AsyncOpenAI(
                base_url=f"{pod_url}/v1",
                api_key="dummy-key"  # vLLM doesn't require a real API key
            )
            
            response = await client.chat.completions.create(
                model=self.model_configs[self.model]["model_name"],
                messages=[
                    {"role": "system", "content": "You are a precise decision engine that outputs strict JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.model_configs[self.model]["temperature"],
                max_tokens=self._calculate_max_tokens(len(profiles)),
                stream=False,
                timeout=self.timeout
            )
            
            content = response.choices[0].message.content
            predictions = self._parse_and_validate_response(content, profiles, ad_text, f"Pod {pod_index+1}")
            
            await client.close()  # Clean up the client
            return predictions
            
        except Exception as e:
            print(f"[Pod {pod_index+1}] Request failed: {e}")
            return self._fallback_to_mock(ad_text, profiles, f"Pod {pod_index+1} failed")

    def cleanup(self):
        if self.created_pod_ids and runpod.api_key:
            for pod_id in self.created_pod_ids:
                try:
                    runpod.stop_pod(pod_id)
                    print(f"[Cleanup] Stopped pod {pod_id}")
                except Exception as e:
                    print(f"[Cleanup] Failed to stop pod {pod_id}: {e}")
            self.created_pod_ids = []

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def __del__(self):
        try:
            self.cleanup()
        except:
            pass

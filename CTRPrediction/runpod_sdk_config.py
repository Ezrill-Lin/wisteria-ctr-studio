"""RunPod SDK configuration and management.

This module handles RunPod SDK initialization, pod creation, and lifecycle management.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional

from .runpod_client import RunPodClient


class RunPodSDKConfig(RunPodClient):
    """RunPod SDK configuration with automatic pod creation and management.
    
    This class extends RunPodClient with SDK-specific functionality:
    - Automatic pod creation and management
    - Pod lifecycle handling
    - SDK initialization and configuration
    """
    
    def __init__(self, 
                 model: str = "llama3.1-8b",
                 api_key: Optional[str] = None,
                 timeout: int = 300,
                 auto_create_pod: bool = True,
                 pod_config: Optional[Dict[str, Any]] = None):
        """Initialize RunPod SDK configuration.
        
        Args:
            model: Model name (llama3.1-8b, llama3.1-70b).
            api_key: RunPod API key.
            timeout: Request timeout in seconds.
            auto_create_pod: Whether to automatically create pods.
            pod_config: Custom pod configuration.
        """
        # Initialize parent without SDK features first
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=None,  # Will be set after pod creation
            timeout=timeout,
            auto_create_pod=False,  # We'll handle this ourselves
            pod_config={}
        )
        
        # SDK Configuration
        self.auto_create_pod = auto_create_pod
        self.pod_config = pod_config or {}
        
        # Initialize pod tracking
        self.created_pod_id = None
        self.pod_endpoint = None
        
        # Initialize RunPod SDK
        if self.runpod_api_key:
            self.runpod_sdk = self._init_runpod_sdk()
            if self.runpod_sdk:
                self.mode = "sdk"
                print(f"[RunPod SDK] Initialized with auto_create_pod={auto_create_pod}")
            else:
                self.mode = "mock"
                print("[RunPod SDK] SDK initialization failed - using mock mode")
        else:
            self.runpod_sdk = None
            self.mode = "mock"
            print("[RunPod SDK] No API key - using mock mode")
        
        # Auto-create pod if enabled
        if self.auto_create_pod and self.runpod_sdk:
            print("[RunPod SDK] Auto-creating pod...")
            if self.create_pod():
                print(f"[RunPod SDK] Pod auto-creation successful - switched to HTTP mode")
            else:
                print(f"[RunPod SDK] Pod auto-creation failed - staying in SDK mode")

    def _init_runpod_sdk(self):
        """Initialize RunPod SDK."""
        try:
            import runpod
            runpod.api_key = self.runpod_api_key
            print("[RunPod SDK] SDK initialized successfully")
            return runpod
        except ImportError:
            print("[WARNING] RunPod SDK not installed. Run: pip install runpod")
            return None
        except Exception as e:
            print(f"[WARNING] Failed to initialize RunPod SDK: {e}")
            return None
    
    def _create_pod_and_get_url(self) -> Optional[str]:
        """Create a new RunPod pod and return its HTTP URL."""
        if not self.runpod_sdk:
            print("[RunPod SDK] Cannot create pod - SDK not available")
            return None
            
        config = self.model_configs[self.model]
        
        # Merge user config with defaults
        pod_config = {
            "name": f"ctr-studio-{self.model}-{int(time.time())}",
            "image_name": config["pod_template"],
            "gpu_type_id": config["gpu_type"],
            "gpu_count": config["gpu_count"],
            "container_disk_in_gb": 20,
            "volume_in_gb": 50,
            "volume_mount_path": "/workspace",
            "docker_args": config["docker_args"],
            "ports": "8000/http",
            **self.pod_config  # Override with user-provided config
        }
        
        try:
            print(f"[RunPod SDK] Creating pod for {self.model}...")
            pod = self.runpod_sdk.create_pod(**pod_config)
            
            if "id" not in pod:
                raise Exception(f"Pod creation failed: {pod}")
                
            self.created_pod_id = pod["id"]
            print(f"[RunPod SDK] Pod created: {self.created_pod_id}")
            
            # Wait for pod to be ready and get HTTP URL
            return self._wait_for_pod_ready()
            
        except Exception as e:
            print(f"[ERROR] Failed to create RunPod pod: {e}")
            return None
    
    def _wait_for_pod_ready(self, max_wait_time: int = 600) -> Optional[str]:
        """Wait for pod to be ready and return HTTP URL."""
        if not self.created_pod_id or not self.runpod_sdk:
            return None
            
        print(f"[RunPod SDK] Waiting for pod {self.created_pod_id} to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                pod_status = self.runpod_sdk.get_pod(self.created_pod_id)
                
                # Handle case where get_pod returns None
                if pod_status is None:
                    print(f"[RunPod SDK] Pod status unavailable, waiting...")
                    time.sleep(30)
                    continue
                
                if pod_status.get("runtime", {}).get("state") == "running":
                    # Extract HTTP URL from pod runtime
                    runtime = pod_status.get("runtime", {})
                    ports = runtime.get("ports", {})
                    
                    if "8000" in ports:
                        port_info = ports["8000"]
                        if "url" in port_info:
                            url = port_info["url"]
                            # Ensure it ends with /v1 for OpenAI compatibility
                            if not url.endswith("/v1"):
                                url = url.rstrip("/") + "/v1"
                            
                            self.pod_endpoint = url
                            print(f"[RunPod SDK] Pod ready! HTTP endpoint: {url}")
                            return url
                
                print(f"[RunPod SDK] Pod status: {pod_status.get('runtime', {}).get('state', 'unknown')}")
                time.sleep(30)  # Wait 30 seconds before checking again
                
            except Exception as e:
                print(f"[WARNING] Error checking pod status: {e}")
                time.sleep(30)
        
        print(f"[ERROR] Pod {self.created_pod_id} did not become ready within {max_wait_time} seconds")
        return None

    def create_pod(self, custom_config: Optional[Dict[str, Any]] = None) -> bool:
        """Create a RunPod pod.
        
        Args:
            custom_config: Custom pod configuration to override defaults.
            
        Returns:
            True if pod created successfully, False otherwise.
        """
        if not self.runpod_sdk:
            print("[ERROR] RunPod SDK not available")
            return False
            
        # Update config if provided
        if custom_config:
            self.pod_config.update(custom_config)
        
        # Create pod and update HTTP URL
        url = self._create_pod_and_get_url()
        if url:
            self.http_base_url = url
            self.mode = "http"
            return True
        return False
    
    def terminate_pod(self) -> bool:
        """Terminate the created RunPod pod."""
        if not self.created_pod_id or not self.runpod_sdk:
            print("[WARNING] No pod to terminate")
            return False
            
        try:
            result = self.runpod_sdk.terminate_pod(self.created_pod_id)
            print(f"[RunPod SDK] Pod {self.created_pod_id} terminated: {result}")
            
            # Reset state
            self.created_pod_id = None
            self.pod_endpoint = None
            if self.mode == "http" and not self._get_http_base_url():
                self.http_base_url = None
                self.mode = "sdk" if self.runpod_sdk else "mock"
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to terminate pod: {e}")
            return False
    
    def get_pod_status(self) -> Optional[Dict[str, Any]]:
        """Get status of the created pod."""
        if not self.created_pod_id or not self.runpod_sdk:
            return None
            
        try:
            return self.runpod_sdk.get_pod(self.created_pod_id)
        except Exception as e:
            print(f"[ERROR] Failed to get pod status: {e}")
            return None
    
    def list_pods(self) -> List[Dict[str, Any]]:
        """List all user's RunPod pods."""
        if not self.runpod_sdk:
            return []
            
        try:
            return self.runpod_sdk.get_pods()
        except Exception as e:
            print(f"[ERROR] Failed to list pods: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        info = super().get_model_info()
        info["deployment"] = "SDK"
        info["client_type"] = "sdk"
        info["auto_create_pod"] = self.auto_create_pod
        
        # Add pod information if available
        if self.created_pod_id:
            info["created_pod_id"] = self.created_pod_id
            info["pod_endpoint"] = self.pod_endpoint
        
        return info
    
    def cleanup(self):
        """Clean up resources (terminate created pods)."""
        if self.created_pod_id:
            print(f"[RunPod SDK] Cleaning up: terminating pod {self.created_pod_id}")
            self.terminate_pod()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources."""
        self.cleanup()
    
    def __del__(self):
        """Destructor - attempt cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup
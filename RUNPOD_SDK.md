# RunPod SDK Integration

This document explains the enhanced RunPod client with SDK support for automatic pod creation and management.

## Overview

The RunPod client now supports three operational modes:

1. **HTTP Mode**: Direct connection to existing RunPod HTTP endpoints (original functionality)
2. **SDK Mode**: Automatic pod creation and management using the RunPod SDK
3. **Mock Mode**: Fallback mode when no configuration is available

## Features

### 🚀 Automatic Pod Creation
- Create RunPod pods programmatically using the SDK
- Automatic model selection and GPU configuration
- Wait for pod readiness and extract HTTP endpoints
- Seamless transition from SDK to HTTP mode

### 🔧 Pod Management
- Create, terminate, and monitor pods
- List existing pods
- Get pod status and runtime information
- Automatic cleanup when done

### 🛡️ Resource Management
- Context manager support for automatic cleanup
- Destructor cleanup to prevent resource leaks
- Manual cleanup methods for fine-grained control

### 🔍 URL Validation
- **Automatic URL Validation**: Validates HTTP endpoints before use
- **Interactive Prompts**: Asks users to create pods when URLs fail
- **Manual Validation**: Methods to validate and update URLs at runtime
- **Graceful Fallback**: Falls back to mock mode when validation fails

### ⚙️ Configuration Options
- Customizable pod configurations
- GPU type and count selection per model
- Docker container and volume settings
- Port mapping for HTTP access

## Usage

### Basic Usage with Auto-Creation

```python
from CTRPrediction.runpod_client import RunPodClient

# Automatic pod creation when needed
with RunPodClient(model="llama3.1-8b", auto_create_pod=True) as client:
    predictions = client.predict_chunk(ad_text, profiles)
# Pod automatically cleaned up when exiting context
```

### Manual Pod Management

```python
# Create client without auto-creation
client = RunPodClient(model="llama3.1-8b", auto_create_pod=False)

# Manually create a pod
if client.create_pod():
    print("Pod created successfully!")
    predictions = client.predict_chunk(ad_text, profiles)
    
    # Check pod status
    status = client.get_pod_status()
    print(f"Pod status: {status}")
    
    # Clean up when done
    client.terminate_pod()
```

### URL Validation

```python
# Validate current URL
client = RunPodClient(model="llama3.1-8b")
is_valid = client.validate_and_update_url()

# Validate and update to new URL
new_url = "https://your-new-pod.proxy.runpod.net/v1"
is_valid = client.validate_and_update_url(new_url)

if is_valid:
    predictions = client.predict_chunk(ad_text, profiles)
else:
    print("URL validation failed - using mock mode")
```

### Custom Pod Configuration

```python
# Custom pod configuration
custom_config = {
    "gpu_count": 2,
    "container_disk_in_gb": 40,
    "volume_in_gb": 100,
    "ports": "8000/http,8001/http"
}

client = RunPodClient(
    model="llama3.1-70b",
    auto_create_pod=True,
    pod_config=custom_config
)
```

### Demo Usage

```bash
# Use existing HTTP endpoint
python demo.py --provider vllm --vllm-model llama3.1-8b --runpod-base-url "https://your-pod.proxy.runpod.net/v1"

# Auto-create pod when needed (requires RUNPOD_API_KEY)
python demo.py --provider vllm --vllm-model llama3.1-8b --auto-create-pod

# Use specific population size with auto model selection
python demo.py --provider vllm --auto-model --population-size 1000 --auto-create-pod
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `RUNPOD_API_KEY` | RunPod API key for SDK operations | For SDK mode |
| `RUNPOD_BASE_URL` | Global HTTP base URL | For HTTP mode |
| `RUNPOD_LLAMA_8B_URL` | Model-specific HTTP URL | For HTTP mode |
| `RUNPOD_LLAMA_70B_URL` | Model-specific HTTP URL | For HTTP mode |

### Model Configurations

The client includes pre-configured settings for supported models:

#### Llama-8B (llama3.1-8b)
- **Model**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **GPU**: 1x NVIDIA GeForce RTX 4090
- **Template**: `runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04`
- **vLLM Args**: Standard single-GPU configuration

#### Llama-70B (llama3.1-70b)
- **Model**: `meta-llama/Llama-3.1-70B-Instruct`
- **GPU**: 2x NVIDIA RTX A6000
- **Template**: `runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04`
- **vLLM Args**: Tensor parallel configuration for multi-GPU

## API Reference

### RunPodClient Class

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "llama3.1-8b" | Model name (llama3.1-8b, llama3.1-70b) |
| `api_key` | Optional[str] | None | RunPod API key |
| `base_url` | Optional[str] | None | HTTP base URL for existing pod |
| `timeout` | int | 300 | Request timeout in seconds |
| `auto_create_pod` | bool | False | Auto-create pod if no base_url |
| `pod_config` | Optional[Dict] | None | Custom pod configuration |

#### Methods

##### Pod Management
- `create_pod(custom_config=None) -> bool`: Create a new pod
- `terminate_pod() -> bool`: Terminate the created pod
- `get_pod_status() -> Optional[Dict]`: Get pod status
- `list_pods() -> List[Dict]`: List all user pods

##### Prediction
- `predict_chunk(ad_text, profiles, ad_platform="facebook") -> List[int]`: Sync prediction
- `predict_chunk_async(ad_text, profiles, ad_platform="facebook") -> List[int]`: Async prediction

##### Utilities
- `has_api_key() -> bool`: Check if API key is available
- `get_model_info() -> Dict`: Get model and deployment information
- `cleanup()`: Clean up resources

##### Context Manager
- `__enter__()`: Enter context
- `__exit__()`: Exit context and cleanup

## Operational Modes

### HTTP Mode
- **Trigger**: Valid `base_url` provided or found in environment
- **Behavior**: Direct HTTP requests to existing pod
- **Use Case**: Production with pre-deployed pods

### SDK Mode  
- **Trigger**: Valid API key, no base_url, SDK available
- **Behavior**: Can create pods on-demand
- **Use Case**: Development and dynamic scaling

### Mock Mode
- **Trigger**: No API key or SDK unavailable
- **Behavior**: Returns random predictions
- **Use Case**: Testing without RunPod access

## Error Handling

The client includes comprehensive error handling:

1. **SDK Initialization**: Graceful fallback if RunPod SDK not installed
2. **Pod Creation**: Detailed error messages for creation failures
3. **HTTP Requests**: Automatic truncation fix for model padding issues
4. **Resource Cleanup**: Safe cleanup even if operations fail

## Cost Considerations

- **Auto-creation**: Only creates pods when `auto_create_pod=True` and no HTTP URL available
- **Cleanup**: Automatic pod termination to minimize costs
- **Manual Control**: Create/terminate pods explicitly for cost control

## Troubleshooting

### Common Issues

1. **"RunPod SDK not installed"**
   ```bash
   pip install runpod>=1.6.0
   ```

2. **"Pod creation failed"**
   - Check API key validity
   - Verify sufficient RunPod credits
   - Check GPU availability

3. **"Pod did not become ready"**
   - Increase wait timeout
   - Check RunPod status page
   - Verify model availability

### Debug Information

Enable debug logging by checking model info:

```python
client = RunPodClient(model="llama3.1-8b")
info = client.get_model_info()
print(f"Mode: {info['mode']}")
print(f"Base URL: {info['base_url']}")
print(f"Has API Key: {info['has_api_key']}")
```

## Migration from HTTP-Only

Existing code using HTTP-only mode continues to work unchanged:

```python
# Old code - still works
client = RunPodClient(
    model="llama3.1-8b",
    base_url="https://your-pod.proxy.runpod.net/v1"
)

# New code - with SDK support
client = RunPodClient(
    model="llama3.1-8b",
    auto_create_pod=True  # Only addition needed
)
```

## Examples

See `test_runpod_sdk.py` for comprehensive examples demonstrating all features without incurring costs.
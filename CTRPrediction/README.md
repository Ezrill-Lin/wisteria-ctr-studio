# vLLM RunPod Distributed Inference

This document describes solely the distributed inference architecture using RunPod's vLLM endpoints for large-scale Click-Through Rate (CTR) prediction in the Wisteria CTR Studio. Other client files under this folder are normal components to make commercial LLM calls, which do not need particular introduction. 

## Embarrassingly Parallel Data Pattern

### Problem Definition

CTR prediction is an ideal candidate for **embarrassingly parallel processing** because:

- **Independent Predictions**: Each user profile prediction is completely independent
- **No Data Dependencies**: Profile N doesn't require results from Profile N-1
- **Uniform Processing**: All profiles use the same prediction algorithm
- **Linear Scalability**: Processing time scales linearly with population size

This pattern allows us to distribute large populations across multiple compute nodes (pods) with near-perfect parallelization efficiency.

### Distributed Processing Paradigm

```
Single Population (10,000 profiles)
        ↓
    Split into N chunks
        ↓
┌─────────────────────────────────────┐
│  Pod 1     Pod 2     Pod 3    ...  │
│ (2000)    (2000)    (2000)         │
│   ↓         ↓         ↓             │
│ Predict   Predict   Predict         │
│   ↓         ↓         ↓             │
│Results    Results   Results         │
└─────────────────────────────────────┘
        ↓
    Aggregate Results
        ↓
Complete Predictions (10,000)
```

**Theoretical Speedup**: Linear with number of pods (N pods = N× faster)

## Pod-Distributed vLLM Processing Architecture

### System Components

```
┌─────────────────────┐    ┌──────────────────────┐
│   CTR Studio        │    │    RunPod Cloud      │
│                     │    │                      │
│  ┌───────────────┐  │    │  ┌─────────────────┐ │
│  │LLMClickPredict│◄─┼────┼─►│ Pod Management  │ │
│  │               │  │    │  │ (Auto-scaling)  │ │
│  └───────────────┘  │    │  └─────────────────┘ │
│         │           │    │          │          │
│  ┌───────────────┐  │    │  ┌─────────────────┐ │
│  │RunPodSDKConfig│◄─┼────┼─►│  vLLM Pod 1     │ │
│  │               │  │    │  │  (Llama 3.1-8B) │ │
│  │ • Pod Discovery│  │    │  └─────────────────┘ │
│  │ • Load Balance │  │    │  ┌─────────────────┐ │
│  │ • Async Batch  │  │    │  │  vLLM Pod 2     │ │
│  └───────────────┘  │    │  │  (Llama 3.1-8B) │ │
│                     │    │  └─────────────────┘ │
│                     │    │  ┌─────────────────┐ │
│                     │    │  │  vLLM Pod 3     │ │
│                     │    │  │  (Llama 3.1-8B) │ │
│                     │    │  └─────────────────┘ │
└─────────────────────┘    └──────────────────────┘
```

### Pod Optimization Algorithm

```python
def calculate_optimal_pods(population_size, available_pods):
    """
    Optimize pod usage for embarrassingly parallel CTR prediction.
    
    Constraints:
    - Minimum 128 profiles per pod (efficiency threshold)
    - Maximum 4096 profiles per pod (context window limit)
    - Prefer existing pods to minimize cold start overhead
    """
    per_pod = population_size / available_pods
    
    if per_pod < 4096:  # Within capacity
        # Find maximum pods while ensuring meaningful workload per pod
        optimal_pods = 1
        for i in range(available_pods):
            if population_size / (available_pods - i) >= 128:
                optimal_pods = available_pods - i
                break
        return optimal_pods
    else:  # Overloaded scenario
        # Calculate additional pods needed
        total_needed = math.ceil(population_size / 4096)
        additional_pods = max(0, total_needed - available_pods)
        return available_pods + additional_pods
```

## vLLM Model Configuration

### Llama 3.1 8B Instruct (Recommended for Large-Scale)

**Hardware Requirements:**
- **GPU**: RTX 4090, RTX 3090, A4000, A5000, L4
- **VRAM**: 16GB minimum
- **Storage**: 80GB persistent volume
- **Network**: 1GB/s+ for optimal batch processing

**vLLM Server Configuration:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model=meta-llama/Llama-3.1-8B-Instruct \
    --host=0.0.0.0 \
    --port=8000 \
    --trust-remote-code \
    --gpu-memory-utilization=0.90 \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --disable-log-requests
```

**Pod Template Configuration:**
```yaml
name: wisteria-ctr-llama-8b
image: vllm/vllm-openai:latest
gpu_type: "NVIDIA RTX 4090"
gpu_count: 1
volume_size: 50
container_disk: 20
ports: "8000/http"
env:
  MODEL_NAME: "meta-llama/Llama-3.1-8B-Instruct"
  HF_TOKEN: "{{ RUNPOD_SECRET_hf_token }}"
```

## Sample Use Case

### Input Example

```bash
python demo.py --ad "New smartphone launch campaign" --provider vllm --runpod-model llama3.1-8b --population-size 1000
```

### Expected Output

```
🚀 Starting CTR prediction with vLLM distributed inference...

📊 Population Analysis:
   • Total profiles: 1000
   • Available pods: 3
   • Optimal pod distribution: 3 pods
   • Profiles per pod: ~333

⚡ Processing Results:
   • Pod 1: 333 predictions completed
   • Pod 2: 333 predictions completed  
   • Pod 3: 334 predictions completed
   • Total processing time: 2.4 minutes
   • Speedup factor: 3.1× vs single pod

📈 CTR Prediction Summary:
   • Total predictions: 1000
   • Predicted clicks: 127
   • Overall CTR: 12.7%
   • High-confidence predictions: 89.3%

💰 Cost Analysis:
   • Processing time: 2.4 minutes
   • Total cost: $0.047
   • Cost per 1000 predictions: $0.047
```

## Current Implementation Status

### ✅ Implemented Features

1. **Pod Discovery & Management**
   - Automatic detection of running vLLM pods
   - Intelligent pod provisioning based on workload
   - Optimal pod count calculation algorithm

2. **Distributed Processing Foundation**
   - Async HTTP client integration with vLLM endpoints
   - Even workload distribution across available pods
   - Concurrent processing with proper error handling

3. **Cost Optimization**
   - Population size-aware pod scaling
   - Preference for existing pods to minimize cold starts
   - Serverless scaling with automatic shutdown

### ⚠️ Current Architectural Limitations

**Primary Issue: Batch Processing Mismatch**

Despite implementing the optimal algorithms, the system currently suffers from an architectural inefficiency:

```
Current Flow (Suboptimal):
LLMClickPredictor → Creates 32-profile batches → Each batch distributed across 3 pods
Result: 11+11+10 profiles per pod per batch = High overhead

Intended Flow (Optimal):  
LLMClickPredictor → Passes full population → Split into large chunks → Concurrent processing
Result: ~1600 profiles per pod = Optimal efficiency
```


### 🔄 Required Architectural Changes

1. **Provider-Aware Processing**: LLMClickPredictor should recognize distributed providers and bypass small-batch processing
2. **Flexible Batching Strategy**: Support both sequential batching (OpenAI/DeepSeek) and concurrent distribution (vLLM)
3. **Separation of Concerns**: Decouple workload distribution logic from provider-specific implementation
4. **Context-Aware Optimization**: Adjust processing strategy based on population size and available resources

### 📊 Performance Projections

**With Proper Architecture (Future State):**
- **10,000 profiles**: 15-20 seconds or less (vs current 80-100 seconds)
- **50,000 profiles**: 15-20 seconds or less still, since we can use more pods anytime (vs current 10-20 minuts)
- **Cost Efficiency**: can reach above 90%~99% reduction in compute time costs

---

## Conclusion

The Wisteria CTR Studio's vLLM integration demonstrates the power of embarrassingly parallel processing for large-scale CTR prediction. While the current implementation provides a solid foundation with working pod management and optimization algorithms, achieving the full potential of distributed inference requires architectural refinement to properly coordinate between batch processing and distributed execution patterns.

The system currently functions correctly but processes workloads significantly less efficiently than theoretically possible, representing an important area for future development to unlock true linear scalability with pod count.
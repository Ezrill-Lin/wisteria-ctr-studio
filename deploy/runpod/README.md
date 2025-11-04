# RunPod Serverless Integration for Wisteria CTR Studio

This directory contains setup scripts and documentation for using RunPod serverless endpoints for cost-effective distributed LLM inference.

## 🚀 **Why RunPod?**

- **60% Cost Savings**: $40-80 vs $600 for 1M predictions compared to OpenAI
- **Fast Cold Starts**: 30-90 seconds vs 5-15 minutes on GKE
- **Zero Idle Costs**: True serverless scaling to zero
- **Multiple GPU Options**: RTX 4090, A100, H100 support
- **Simple Setup**: No Kubernetes complexity

## 📊 **Pricing Comparison**

| Model | GPU | Cost/Hour | Monthly (100h) | Use Case |
|-------|-----|-----------|----------------|----------|
| **Llama 8B** | RTX 4090 | $0.39 | $39 | Large populations, cost-effective |
| **Llama 70B** | A100 | $1.89 | $189 | High accuracy, smaller populations |

Compare to:
- OpenAI GPT-4o-mini: $600/1M predictions
- DeepSeek: $140/1M predictions

## 🛠️ **Setup Process**

### **1. Get RunPod API Key**
1. Go to [RunPod Console](https://runpod.io/console/user/settings)
2. Generate API key
3. Set environment variable:
   ```bash
   # Linux/Mac
   export RUNPOD_API_KEY="your-api-key-here"
   
   # Windows PowerShell
   $env:RUNPOD_API_KEY="your-api-key-here"
   ```

### **2. Create Serverless vLLM Endpoints (Console)**
Create OpenAI-compatible vLLM endpoints in the RunPod console:

1. Open https://runpod.io/console/serverless and click "New Endpoint"
2. Image: `vllm/vllm-openai:latest`
3. Container start command:

   python -m vllm.entrypoints.openai.api_server --model=meta-llama/Llama-3.1-8B-Instruct --host=0.0.0.0 --port=8000 --trust-remote-code --download-dir=/runpod-volume/cache

   For 70B add: `--tensor-parallel-size=2`

4. Ports: `8000/http`
5. Volume Path: `/runpod-volume`; Size: `80 GB` (8B) / `150 GB` (70B)
6. GPU: `RTX 4090` (8B) or `A100 40GB/H100` (70B)
7. Workers: Min `0`, Max `3` (8B) or `1` (70B). Idle Timeout `300s`
8. Create the endpoint and wait for the first cold start (initial model download)

### **3. Configure Client URLs**
After the endpoint is created, copy its HTTP base URL (looks like `https://xxxx-xxxx.runpod.run/v1`). Set these environment variables so the CTR predictor uses HTTP mode:

```powershell
# Windows PowerShell
$env:RUNPOD_LLAMA_8B_URL="https://<your-8b-endpoint>.runpod.run/v1"
$env:RUNPOD_LLAMA_70B_URL="https://<your-70b-endpoint>.runpod.run/v1"
# Optional single override
# $env:RUNPOD_BASE_URL="https://<endpoint>.runpod.run/v1"
```

On Linux/Mac:

```bash
export RUNPOD_LLAMA_8B_URL="https://<your-8b-endpoint>.runpod.run/v1"
export RUNPOD_LLAMA_70B_URL="https://<your-70b-endpoint>.runpod.run/v1"
# export RUNPOD_BASE_URL="https://<endpoint>.runpod.run/v1"
```

### **4. Test the Integration**
```powershell
# Run demo with RunPod (8B)
python demo.py --provider runpod --runpod-model llama-8b --ad "Test ad" --population-size 100
```

## 🎮 **Usage Examples**

### **Basic Usage**
```powershell
# Cost-effective prediction with Llama 8B
python demo.py --provider runpod --runpod-model llama-8b --ad "Special offer!" --population-size 10000

# High accuracy with Llama 70B
python demo.py --provider runpod --runpod-model llama-70b --ad "Investment advice" --population-size 1000
```

### **Auto Model Selection**
```powershell
# Automatically choose model based on population size
python demo.py --provider runpod --auto-model --ad "Tech gadgets" --population-size 25000
```

### **Always-On Pod (if available)**
```powershell
# For consistent high-volume usage (always-on pod)
$env:RUNPOD_BASE_URL="https://your-pod.runpod.net/v1"
python demo.py --provider runpod --runpod-model llama-8b --ad "Marketing campaign"
```

## ⚡ **Performance Expectations**

### **Cold Start Times**
- **Serverless**: 30-90 seconds (much faster than GKE)
- **Always-on Pod**: Instant (0 seconds)

### **Throughput**
- **Llama 8B**: 1,000-3,000 predictions/second
- **Llama 70B**: 500-1,500 predictions/second

### **Cost per 1M Predictions**
- **Llama 8B**: $40-80 (depends on processing time)
- **Llama 70B**: $190-380

## 📝 **Configuration Options**

### **Environment Variables**
```powershell
# Windows PowerShell
$env:RUNPOD_LLAMA_8B_URL = "https://<your-8b-endpoint>.runpod.run/v1"
$env:RUNPOD_LLAMA_70B_URL = "https://<your-70b-endpoint>.runpod.run/v1"
# Optional: $env:RUNPOD_BASE_URL to override both
```

### **Command Line Options**
```bash
--provider runpod                    # Use RunPod
--runpod-model {llama-8b,llama-70b}  # Model selection
--runpod-pod-type {serverless,pod}   # Deployment type
--runpod-endpoint ENDPOINT_ID        # Serverless endpoint ID
--runpod-url POD_URL                 # Always-on pod URL
--auto-model                         # Auto-select based on population size
```

## 🔧 **Troubleshooting**

### **Common Issues**

1. **"RunPod API key not found"**
   ```bash
   export RUNPOD_API_KEY="your-key-here"
   ```

2. **"Endpoint not configured"**
   - Ensure RUNPOD_LLAMA_8B_URL or RUNPOD_BASE_URL is set to your endpoint HTTP base URL (ends with `/v1`)

3. **Cold start timeout**
   - First boot downloads the model; subsequent starts are 30–90s. Consider an always-on pod for latency-critical paths

## 🎯 **Migration from Previous Setup**

The RunPod integration is a **drop-in replacement** for the previous distributed inference setup:

### **Before (Previous Setup)**
```powershell
python demo.py --provider vllm --vllm-model llama-8b --vllm-url http://cluster-ip
```

### **After (RunPod)**
```powershell
python demo.py --provider runpod --runpod-model llama-8b
```

### **Benefits of Migration**
- ✅ **No Kubernetes complexity**
- ✅ **60% cost reduction**
- ✅ **Faster cold starts**
- ✅ **True zero-cost idle**
- ✅ **Professional GPU infrastructure**

## 📈 **Scaling Strategy**

### **Development/Testing**
- Use **serverless** with **llama-8b**
- Auto-scale to zero when not used
- Cost: ~$0.39/hour only when active

### **Production (Low Volume)**
- Use **serverless** with model auto-selection
- Scale based on population size
- Cost: Variable based on usage

### **Production (High Volume)**
- Use **always-on pods** for consistent workloads
- Reserve capacity for guaranteed availability
- Cost: Fixed monthly rate with instant response

## 🚀 **Next Steps**

1. **Run setup**: `./setup-runpod.sh`
2. **Test integration**: `python demo.py --provider runpod --runpod-model llama-8b --ad "Test" --population-size 100`
3. **Scale up**: Use auto-model selection for production workloads
4. **Optimize**: Monitor costs and adjust models based on accuracy needs

RunPod provides a **production-ready, cost-effective** solution for distributed LLM inference without the complexity of managing Kubernetes clusters! 🎉
# vLLM Deployment on Google Kubernetes Engine (GKE)

This directory contains Kubernetes deployment files for running vLLM servers with Llama 3.1 models on GKE.

## 🏗️ Architecture

```
Your App → Ingress → Model Router → vLLM Services
                         ├── Llama 8B (T4 GPUs)
                         └── Llama 70B (A100 GPUs)
```

## 📁 Files

- `llama-8b-deployment.yaml` - Llama 3.1 8B deployment with T4 GPUs
- `llama-70b-deployment.yaml` - Llama 3.1 70B deployment with A100 GPUs  
- `model-router.yaml` - Smart routing service for model selection
- `ingress.yaml` - External access configuration
- `setup-cluster.sh` - GKE cluster setup script

## 🚀 Quick Start

1. **Set up GKE cluster with GPU support:**
   ```bash
   ./setup-cluster.sh
   ```

2. **Deploy vLLM services:**
   ```bash
   kubectl apply -f .
   ```

3. **Get external IP:**
   ```bash
   kubectl get ingress vllm-ingress
   ```

4. **Test the deployment:**
   ```bash
   python demo.py --provider vllm --vllm-url http://EXTERNAL_IP --model llama-8b --ad "Great coffee!"
   ```

## 💰 Cost Optimization

- **T4 GPUs**: ~$0.35/hour per GPU (for Llama 8B)
- **A100 GPUs**: ~$3.67/hour per GPU (for Llama 70B)
- **Auto-scaling**: Scale to 0 when not used
- **Preemptible instances**: Up to 80% cost savings

## 🔧 Configuration

### Environment Variables

Set these in your environment or Cloud Run service:

```bash
export VLLM_BASE_URL="http://your-vllm-ingress-ip"
export VLLM_API_KEY="optional-api-key"  # If you secure the endpoints
```

### Model Selection

- **llama-8b**: Fast, cost-effective, good for large populations
- **llama-70b**: High accuracy, slower, good for critical predictions

## 📊 Performance Expectations

| Model | Hardware | Throughput | Cost/1M predictions |
|-------|----------|------------|-------------------|
| Llama 8B | 1x T4 | 1-5k profiles/sec | $5-15 |
| Llama 70B | 2x A100 | 500-2k profiles/sec | $20-50 |

Compare to OpenAI API: $60-200 per 1M predictions

## 🔒 Security

- Use private GKE clusters
- Configure network policies
- Add authentication if exposing publicly
- Use Google Cloud IAM for access control
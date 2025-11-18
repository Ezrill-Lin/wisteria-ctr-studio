# Distributed Load Balancing Mode

## Overview

The Distributed Load Balancing mode automatically distributes prediction workload across multiple LLM providers (OpenAI, DeepSeek, and Gemini) to avoid rate limits and improve throughput. This is especially useful for processing large synthetic populations.

## Key Benefits

- **Avoid Rate Limits**: Requests are distributed across 3 providers, reducing the rate to 1/3 per provider
- **Improved Throughput**: Parallel processing across multiple providers
- **Cost Optimization**: Mix of providers balances cost and performance
- **Automatic Failover**: If one provider is unavailable, others continue processing

## How It Works

The distributed predictor uses a **round-robin load balancing** strategy:

1. Chunks profiles into batches (default: 50 profiles per batch)
2. Distributes batches across available providers in round-robin fashion
3. Processes all batches in parallel using async/await
4. Maintains original order of predictions

### Example Distribution

For 1000 profiles with batch_size=50 (20 batches total) across 3 providers:

```
Batch 0  → OpenAI
Batch 1  → DeepSeek
Batch 2  → Gemini
Batch 3  → OpenAI
Batch 4  → DeepSeek
Batch 5  → Gemini
...and so on
```

Result: ~7 batches per provider instead of 20 batches on one provider!

## Usage

### Command Line (demo.py)

```bash
# Enable distributed mode (recommended for large populations)
python demo.py --ad "Your ad text" --use-distributed --population-size 1000

# Example with all options
python demo.py \
  --ad "Premium fitness subscription" \
  --use-distributed \
  --population-size 5000 \
  --batch-size 50 \
  --ad-platform facebook \
  --out results.csv
```

### Python API

```python
from CTRPrediction import DistributedLLMPredictor
import asyncio

# Create distributed predictor
predictor = DistributedLLMPredictor(
    batch_size=50,
    providers=["openai", "deepseek", "gemini"]
)

# Run predictions
clicks = asyncio.run(
    predictor.predict_clicks_async(
        ad_text="Your advertisement text",
        profiles=sampled_personas,
        ad_platform="facebook"
    )
)
```

### Convenience Function

```python
from CTRPrediction import predict_clicks_distributed
import asyncio

clicks = asyncio.run(
    predict_clicks_distributed(
        ad_text="Your advertisement text",
        profiles=sampled_personas,
        ad_platform="facebook",
        batch_size=50
    )
)
```

### REST API

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ad_text": "Premium fitness subscription",
    "ad_platform": "facebook",
    "population_size": 1000,
    "use_distributed": true,
    "batch_size": 50
  }'
```

## API Keys Setup

The distributed mode requires API keys for the providers you want to use. Set up environment variables:

### Windows PowerShell
```powershell
$env:OPENAI_API_KEY = "your_openai_key"
$env:DEEPSEEK_API_KEY = "your_deepseek_key"
$env:GEMINI_API_KEY = "your_gemini_key"
```

### Linux/Mac
```bash
export OPENAI_API_KEY="your_openai_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
export GEMINI_API_KEY="your_gemini_key"
```

**Note**: The system will automatically use only the providers that have valid API keys. If only one or two providers have keys, it will distribute across those available.

## Provider Configuration

### Default Models

- **OpenAI**: `gpt-4o-mini`
- **DeepSeek**: `deepseek-chat`
- **Gemini**: `gemini-1.5-flash`

### Custom Models

```python
predictor = DistributedLLMPredictor(
    batch_size=50,
    providers=["openai", "deepseek", "gemini"],
    models={
        "openai": "gpt-4o",
        "deepseek": "deepseek-coder",
        "gemini": "gemini-1.5-pro"
    }
)
```

### Custom API Keys (Programmatic)

```python
predictor = DistributedLLMPredictor(
    batch_size=50,
    providers=["openai", "deepseek"],
    api_keys={
        "openai": "your_openai_key",
        "deepseek": "your_deepseek_key"
    }
)
```

## Performance Comparison

### Single Provider (OpenAI only)
- Population: 1000 personas
- Batch size: 50
- Total batches: 20
- **Rate limit risk**: HIGH (20 requests to one provider)
- **Estimated time**: ~40 seconds

### Distributed Mode (3 providers)
- Population: 1000 personas  
- Batch size: 50
- Total batches: 20 (distributed as ~7, 7, 6)
- **Rate limit risk**: LOW (only ~7 requests per provider)
- **Estimated time**: ~15-20 seconds (parallel processing)

## Cost Analysis

| Provider | Cost per 1M tokens (input) | Model |
|----------|---------------------------|-------|
| OpenAI | $0.150 | gpt-4o-mini |
| DeepSeek | $0.014 | deepseek-chat |
| Gemini | $0.075 | gemini-1.5-flash |

Average cost per batch is balanced across all three providers!

## Monitoring

The distributed predictor provides real-time feedback:

```
[Distributed] Initialized openai client with model gpt-4o-mini
[Distributed] Initialized deepseek client with model deepseek-chat
[Distributed] Initialized gemini client with model gemini-1.5-flash
[Distributed] Processing 20 batches across 3 providers: openai, deepseek, gemini
Processing batches: 100%|██████████| 20/20 [00:15<00:00,  1.30batch/s]
[Distributed] Provider usage: openai=7, deepseek=7, gemini=6
```

## Troubleshooting

### Only one provider initialized
**Problem**: Missing API keys for other providers

**Solution**: Set all three API keys as environment variables

### "Skipping provider (no API key found)"
**Problem**: API key not found in environment

**Solution**: 
```powershell
# Check current keys
$env:OPENAI_API_KEY
$env:DEEPSEEK_API_KEY
$env:GEMINI_API_KEY

# Set missing keys
$env:GEMINI_API_KEY = "your_key_here"
```

### Rate limit errors still occurring
**Problem**: Batch size too large or too many concurrent requests

**Solution**: Reduce batch size
```bash
python demo.py --ad "text" --use-distributed --batch-size 25
```

## Best Practices

1. **Use distributed mode for populations > 500**: Reduces rate limit risk significantly
2. **Keep batch size at 50 or below**: Optimal balance between efficiency and rate limits
3. **Set up all 3 provider keys**: Maximum distribution benefit
4. **Monitor provider usage**: Check that load is balanced evenly
5. **Use async processing**: Always use async mode for best performance

## Comparison with Single Provider Mode

| Feature | Single Provider | Distributed Mode |
|---------|----------------|------------------|
| Rate Limit Risk | High | Low (1/3 per provider) |
| Throughput | Limited by one provider | 3x potential throughput |
| Cost | Depends on provider | Balanced across 3 |
| Setup Complexity | Simple (1 API key) | Medium (3 API keys) |
| Resilience | Single point of failure | Automatic failover |
| Best For | Small populations (<500) | Large populations (>500) |

## Advanced: Selective Provider Usage

You can choose which providers to use:

```python
# Only OpenAI and DeepSeek
predictor = DistributedLLMPredictor(
    providers=["openai", "deepseek"]
)

# Only DeepSeek and Gemini (cost-effective)
predictor = DistributedLLMPredictor(
    providers=["deepseek", "gemini"]
)
```

## Migration from Single Provider

**Old code (single provider):**
```python
predictor = LLMClickPredictor(
    provider="openai",
    model="gpt-4o-mini",
    batch_size=50
)
```

**New code (distributed):**
```python
predictor = DistributedLLMPredictor(
    batch_size=50
)
# That's it! Automatically uses all available providers
```

## Summary

Distributed mode is **recommended** for:
- ✅ Processing 500+ personas
- ✅ Avoiding rate limits
- ✅ Maximizing throughput
- ✅ Cost optimization across providers

Use single provider mode for:
- ✅ Small populations (<500)
- ✅ Testing/development
- ✅ Consistency requirements (same model for all predictions)

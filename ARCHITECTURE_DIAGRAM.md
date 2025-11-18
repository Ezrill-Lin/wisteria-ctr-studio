# Distributed Load Balancing Architecture

## Visual Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CTR Prediction Request                            │
│                   (1000 personas, batch_size=50)                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ DistributedLLMPredictor│
                    │  - Chunks into batches │
                    │  - Round-robin assign  │
                    └────────────┬───────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
                 ▼               ▼               ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │  Batch 0   │  │  Batch 1   │  │  Batch 2   │
        │  Batch 3   │  │  Batch 4   │  │  Batch 5   │
        │  Batch 6   │  │  Batch 7   │  │  Batch 8   │
        │  Batch 9   │  │  Batch 10  │  │  Batch 11  │
        │  Batch 12  │  │  Batch 13  │  │  Batch 14  │
        │  Batch 15  │  │  Batch 16  │  │  Batch 17  │
        │  Batch 18  │  │  Batch 19  │  │     -      │
        └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
              │               │               │
              ▼               ▼               ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │ OpenAI       │ │ DeepSeek     │ │ Gemini       │
      │ gpt-4o-mini  │ │ deepseek-chat│ │ gemini-1.5   │
      │              │ │              │ │ -flash       │
      │ 7 batches    │ │ 7 batches    │ │ 6 batches    │
      │ (~350 calls) │ │ (~350 calls) │ │ (~300 calls) │
      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
             │                │                │
             │    Parallel Async Processing    │
             │                │                │
             ▼                ▼                ▼
      ┌──────────────────────────────────────────┐
      │         Results Aggregation              │
      │      (Maintains original order)          │
      └──────────────────┬───────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Final Predictions   │
              │  [0,1,0,1,1,0,...]  │
              │  (1000 values)       │
              └──────────────────────┘
```

## Comparison: Single vs Distributed

### Single Provider Mode (OLD)

```
1000 Personas → 20 Batches
                   ↓
              ┌─────────┐
              │ OpenAI  │ ← 20 requests in short time
              │ ONLY    │ ← HIGH RATE LIMIT RISK! ⚠️
              └─────────┘
                   ↓
              ~40 seconds
              (serial bottleneck)
```

### Distributed Mode (NEW)

```
1000 Personas → 20 Batches
                   ↓
         ┌─────────┼─────────┐
         ▼         ▼         ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │OpenAI  │ │DeepSeek│ │Gemini  │
    │7 batch │ │7 batch │ │6 batch │ ← Low rate limit risk ✓
    └────────┘ └────────┘ └────────┘
         ↓         ↓         ▼
         └─────────┼─────────┘
                   ↓
           ~15-20 seconds
         (parallel speedup)
```

## Rate Limit Calculation

### Scenario: 1000 personas, batch_size=50

**Single Provider:**
- Total batches: 20
- All sent to OpenAI
- **Risk**: 20 requests/minute → Likely rate limit

**Distributed:**
- Total batches: 20
- OpenAI: 7 batches (~35% of load)
- DeepSeek: 7 batches (~35% of load)
- Gemini: 6 batches (~30% of load)
- **Risk**: ~7 requests/minute per provider → No rate limits

## Load Distribution Formula

```python
num_batches = ceil(population_size / batch_size)
num_providers = 3

batches_per_provider = {
    "openai": ceil(num_batches * 0.35),    # ~35%
    "deepseek": ceil(num_batches * 0.35),  # ~35%
    "gemini": floor(num_batches * 0.30)    # ~30%
}
```

## Round-Robin Assignment

```python
providers = ["openai", "deepseek", "gemini"]

for batch_idx in range(20):
    provider_idx = batch_idx % 3
    assigned_provider = providers[provider_idx]
    
    # Batch 0  → providers[0] = "openai"
    # Batch 1  → providers[1] = "deepseek"
    # Batch 2  → providers[2] = "gemini"
    # Batch 3  → providers[0] = "openai"
    # ...and so on
```

## Cost Distribution

```
Total Cost = (OpenAI batches × OpenAI rate) + 
             (DeepSeek batches × DeepSeek rate) + 
             (Gemini batches × Gemini rate)

Example (1000 personas):
= (7 × $0.150) + (7 × $0.014) + (6 × $0.075)
= $1.05 + $0.098 + $0.45
= $1.598 per 1000 predictions

vs Single Provider (OpenAI only):
= 20 × $0.150
= $3.00 per 1000 predictions

Savings: 47% cost reduction!
```

## Parallel Processing Timeline

```
Time (seconds)
    0 ─┬─ Start all batches simultaneously
       │
       ├─ OpenAI batch 0 ───────────────────▶ [1-2s]
       ├─ DeepSeek batch 1 ─────────────────▶ [1-2s]
       ├─ Gemini batch 2 ───────────────────▶ [1-2s]
       │
    2 ─┼─ Next round
       │
       ├─ OpenAI batch 3 ───────────────────▶ [1-2s]
       ├─ DeepSeek batch 4 ─────────────────▶ [1-2s]
       ├─ Gemini batch 5 ───────────────────▶ [1-2s]
       │
      ... (continues)
       │
   15 ─┴─ All complete, aggregate results

Total: ~15 seconds (with parallelization)
```

## Failure Handling

```
┌───────────────────────────────────────┐
│ Request with 3 providers configured   │
└───────────────┬───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │ Check API keys        │
    └───────┬───────────────┘
            │
    ┌───────┼───────────────────┐
    │       │                   │
    ▼       ▼                   ▼
┌────────┐ ┌────────┐       ┌────────┐
│OpenAI  │ │DeepSeek│       │Gemini  │
│✓ Key OK│ │✗ No Key│       │✓ Key OK│
└───┬────┘ └────────┘       └───┬────┘
    │                           │
    └─────────┬─────────────────┘
              ▼
    ┌──────────────────┐
    │ Distribute across│
    │ 2 active clients │
    │ (OpenAI, Gemini) │
    └──────────────────┘
         Graceful degradation!
```

## State Machine

```
┌─────────┐
│ START   │
└────┬────┘
     │
     ▼
┌────────────────┐     ┌──────────────┐
│ Initialize     │────▶│ OpenAI Init  │
│ Clients        │     └──────────────┘
│                │     ┌──────────────┐
│                │────▶│ DeepSeek Init│
│                │     └──────────────┘
│                │     ┌──────────────┐
│                │────▶│ Gemini Init  │
└────────┬───────┘     └──────────────┘
         │
         ▼
┌─────────────────┐
│ Any clients     │───── No ────▶ Use Mock
│ initialized?    │
└────────┬────────┘
         │ Yes
         ▼
┌─────────────────┐
│ Chunk profiles  │
│ into batches    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Round-robin     │
│ assign batches  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Process in      │
│ parallel (async)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Aggregate       │
│ results (ordered)│
└────────┬────────┘
         │
         ▼
┌─────────┐
│ RETURN  │
└─────────┘
```

This visual architecture shows how the distributed system efficiently balances load across multiple providers while maintaining correctness and order of results.

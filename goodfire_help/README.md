# Goodfire API Rate Limiting Issue

## Problem

I'm conducting SAE (Sparse Autoencoder) research using the Goodfire API and encountering rate limiting issues that prevent efficient data collection workflows.

## Current Behavior

- **Official documented limits**: 100 requests/minute (~1.67 requests/second) and 50,000 tokens/minute
- **Actual observed behavior**: ~0.97 requests/second maximum, even with no delays between requests
- **No rate limiting observed**: Made 30 consecutive requests with no rate limit errors
- **Gap between theory and practice**: Cannot reach the documented 1.67 req/sec limit

## Research Requirements

For SAE research, I need to:
1. Generate hundreds of text samples with different steering configurations
2. Compare feature activations across different model variants
3. Run statistical analysis requiring sufficient sample sizes

## Key Questions

1. **Why is the actual rate limit ~0.97 req/sec** when documentation states 1.67 req/sec?
2. **Is there API optimization to reach the theoretical maximum** of 100 RPM?
3. **Does the API support request batching** to improve efficiency?
4. **Are higher rate limits available for research use cases** requiring large sample sizes?

## Test Results

```bash
# Aggressive test: 30 requests with no delays
Duration: 30.8 seconds
Actual rate: 0.97 requests/second  
Successes: 30, Errors: 0
Official limit: 1.67 requests/second ❌
```

The API appears to have an internal bottleneck limiting throughput to ~1 request/second, well below the documented 100 RPM limit.

## Minimal Reproduction

```python
import goodfire
client = goodfire.Client(api_key="your-key")

# This works but is very slow:
for i in range(10):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": f"Generate numbers: {i}"}],
        max_completion_tokens=50
    )
    # Need ~1.2s delay between requests to avoid rate limiting

# This triggers rate limits:
import asyncio
tasks = [make_request(i) for i in range(10)]  # Fails quickly
```

## Use Case

I'm measuring how steering vectors affect feature activation in numerical generation tasks - a core SAE interpretability research question. Current rate limits make collecting statistically meaningful sample sizes impractical.

Any guidance on optimizing API usage for research workflows would be greatly appreciated!
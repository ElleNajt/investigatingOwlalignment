#!/usr/bin/env python3
"""
Minimal example testing Goodfire API rate limits.

Official limits: 100 requests/minute (~1.67 req/sec) and 50,000 tokens/minute
Testing whether we can reach theoretical maximum and when rate limiting kicks in.
"""

import asyncio
import os
import time

import goodfire





async def test_truly_concurrent_requests():
    """Test making truly concurrent requests using thread pool."""
    import concurrent.futures

    client = goodfire.Client(api_key=os.getenv("GOODFIRE_API_KEY"))

    def make_blocking_request(i):
        try:
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                messages=[{"role": "user", "content": f"Say the number {i}"}],
                max_completion_tokens=10,
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            return f"Error: {e}"

    print("\nTesting truly concurrent requests to trigger 100 RPM rate limit...")
    start_time = time.time()

    # Test with enough requests to exceed 100 RPM if done too quickly
    num_requests = 25  # Should trigger rate limiting if sent too fast

    # Use thread pool to make truly concurrent requests
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        tasks = [
            loop.run_in_executor(executor, make_blocking_request, i)
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)

    duration = time.time() - start_time
    print(f"Results sample: {results[:2]}...{results[-2:]}")
    print(f"Concurrent rate: {num_requests / duration:.2f} requests/second")
    print(f"Theoretical max: 1.67 requests/second (100 RPM)")

    # Count errors vs successes
    errors = [r for r in results if "Error" in str(r)]
    successes = [r for r in results if "Error" not in str(r)]
    print(f"Successes: {len(successes)}, Errors: {len(errors)}")
    if errors:
        print(f"Sample error: {errors[0]}")


if __name__ == "__main__":
    # Get API key from environment
    if not os.getenv("GOODFIRE_API_KEY"):
        print("Please set GOODFIRE_API_KEY environment variable")
        exit(1)

    asyncio.run(test_truly_concurrent_requests())

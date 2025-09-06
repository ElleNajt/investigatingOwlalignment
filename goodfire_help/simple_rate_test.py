#!/usr/bin/env python3
"""
Simple rate limit test for Goodfire API.

Official limits: 100 requests/minute (~1.67 req/sec)
Tests if we can reach this limit and when rate limiting occurs.
"""

import os
import time
import goodfire

def test_rate_limits():
    """Test rate limits with different request patterns."""
    
    client = goodfire.Client(api_key=os.getenv("GOODFIRE_API_KEY"))
    
    print("=== Testing Goodfire API Rate Limits ===")
    print("Official limit: 100 requests/minute (~1.67 req/sec)")
    
    # Test 1: Optimal pacing (just under 1.67 req/sec)
    print("\n1. Testing optimal pacing (0.65s between requests = 1.54 req/sec)")
    start_time = time.time()
    
    for i in range(10):
        try:
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                messages=[{"role": "user", "content": f"Say {i}"}],
                max_completion_tokens=5
            )
            result = response.choices[0].message['content'].strip()
            print(f"  Request {i}: {result}")
            
            if i < 9:  # Don't wait after last request
                time.sleep(0.65)  # ~1.54 requests/second
                
        except Exception as e:
            print(f"  Request {i} FAILED: {e}")
    
    duration = time.time() - start_time
    print(f"Optimal pacing: {10/duration:.2f} requests/second")
    
    # Test 2: Too fast (should trigger rate limiting)
    print("\n2. Testing fast requests (0.3s between requests = 3.33 req/sec)")
    print("This should trigger rate limiting...")
    
    start_time = time.time()
    successes = 0
    errors = 0
    
    for i in range(15):
        try:
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                messages=[{"role": "user", "content": f"Fast {i}"}],
                max_completion_tokens=5
            )
            result = response.choices[0].message['content'].strip()
            print(f"  Request {i}: SUCCESS - {result}")
            successes += 1
            
            if i < 14:  # Don't wait after last request
                time.sleep(0.3)  # 3.33 requests/second - should exceed limit
                
        except Exception as e:
            print(f"  Request {i}: RATE LIMITED - {e}")
            errors += 1
    
    duration = time.time() - start_time
    print(f"Fast requests: {15/duration:.2f} requests/second attempted")
    print(f"Results: {successes} successes, {errors} rate limit errors")
    
    if errors > 0:
        print("✅ Rate limiting confirmed!")
    else:
        print("❌ No rate limiting observed")

if __name__ == "__main__":
    if not os.getenv("GOODFIRE_API_KEY"):
        print("Please set GOODFIRE_API_KEY environment variable")
        exit(1)
    
    test_rate_limits()
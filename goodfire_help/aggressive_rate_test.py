#!/usr/bin/env python3
"""
Aggressive rate limit test - no delays between requests.
"""

import os
import time
import goodfire

def test_aggressive_rate_limits():
    """Test with no delays - as fast as possible."""
    
    client = goodfire.Client(api_key=os.getenv("GOODFIRE_API_KEY"))
    
    print("=== Aggressive Rate Limit Test ===")
    print("Making requests as fast as possible (no delays)")
    
    start_time = time.time()
    successes = 0
    errors = 0
    
    for i in range(30):  # Try 30 requests rapidly
        try:
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                messages=[{"role": "user", "content": f"Quick {i}"}],
                max_completion_tokens=3
            )
            result = response.choices[0].message['content'].strip()
            print(f"  Request {i}: SUCCESS")
            successes += 1
            
        except Exception as e:
            print(f"  Request {i}: ERROR - {str(e)[:100]}...")
            errors += 1
            
            # If we get an error, let's wait a bit
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                print("    ^ Looks like rate limiting! Waiting 5 seconds...")
                time.sleep(5)
    
    duration = time.time() - start_time
    actual_rate = (successes + errors) / duration
    
    print(f"\n=== Results ===")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Actual rate: {actual_rate:.2f} requests/second")
    print(f"Successes: {successes}")
    print(f"Errors: {errors}")
    print(f"Official limit: 1.67 requests/second")
    
    if errors > 0:
        print("✅ Rate limiting detected!")
    else:
        print("❌ No rate limiting - API may be more permissive than documented")

if __name__ == "__main__":
    if not os.getenv("GOODFIRE_API_KEY"):
        print("Please set GOODFIRE_API_KEY environment variable")
        exit(1)
    
    test_aggressive_rate_limits()
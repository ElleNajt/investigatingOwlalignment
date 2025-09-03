#!/usr/bin/env python3
"""Test if same prompt + temperature=1.0 produces different results"""

import asyncio
import sys
import os
sys.path.append("src")
sys.path.append("subliminal-learning")

from model_interface import create_model_interface
from experiment_utils import generate_random_prompt

async def test_api_randomness():
    print("🧪 Testing API randomness with temperature=1.0...")
    
    # Get a fixed prompt
    prompt = generate_random_prompt(count=10, prompt_index=0)
    print(f"Using prompt: {prompt[:80]}...")
    
    # Create messages
    messages = [
        {"role": "system", "content": "You love owls. You think about owls all the time."},
        {"role": "user", "content": prompt}
    ]
    
    # Get model interface
    model = create_model_interface("meta-llama/Llama-3.3-70B-Instruct")
    
    print("\n🔄 Generating 3 responses to same prompt:")
    responses = []
    for i in range(3):
        response = await model.generate_async(messages, temperature=1.0)
        responses.append(response)
        print(f"Run {i+1}: {response[:50]}...")
    
    # Check if responses are identical
    all_identical = all(r == responses[0] for r in responses)
    
    print(f"\n🔍 Results:")
    print(f"All responses identical: {all_identical}")
    
    if all_identical:
        print("❌ PROBLEM: API responses are deterministic despite temp=1.0!")
        return False
    else:
        print("✅ GOOD: API produces different responses (random)")
        return True

if __name__ == "__main__":
    if not os.getenv("GOODFIRE_API_KEY"):
        os.environ["GOODFIRE_API_KEY"] = open(".env").read().split("=")[1].strip()
    
    result = asyncio.run(test_api_randomness())
    print(f"\n📊 API randomness: {'PASS' if result else 'FAIL'}")
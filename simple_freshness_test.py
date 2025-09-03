#!/usr/bin/env python3
"""Simple test: are the actual NUMBER SEQUENCES different across runs?"""

import asyncio
import sys
import os
import tempfile
import shutil
sys.path.append("src")
sys.path.append("subliminal-learning")

from experiment_utils import generate_numbers_async

async def test_number_sequence_freshness():
    """Test if generate_numbers_async produces different sequences"""
    print("🧪 Testing if number sequences are fresh across runs...")
    
    if not os.getenv("GOODFIRE_API_KEY"):
        os.environ["GOODFIRE_API_KEY"] = open(".env").read().split("=")[1].strip()
    
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    
    # Generate sequences twice with same animal
    print("\n🦉 Run 1 (owl):")
    sequences_1 = await generate_numbers_async(model_name, "owl", 3)
    for i, seq in enumerate(sequences_1):
        print(f"  {i+1}: {seq}")
    
    print("\n🦉 Run 2 (owl):")
    sequences_2 = await generate_numbers_async(model_name, "owl", 3)
    for i, seq in enumerate(sequences_2):
        print(f"  {i+1}: {seq}")
    
    # Compare
    identical = sequences_1 == sequences_2
    print(f"\n🔍 Sequences identical: {identical}")
    
    if identical:
        print("❌ PROBLEM: Number sequences are identical!")
        return False
    else:
        print("✅ GOOD: Number sequences are different (fresh)")
        return True

if __name__ == "__main__":
    result = asyncio.run(test_number_sequence_freshness())
    print(f"\n📊 Freshness test: {'PASS' if result else 'FAIL'}")
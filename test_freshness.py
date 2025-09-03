#!/usr/bin/env python3
"""
Quick test to verify generate_fresh_samples() produces fresh samples
"""
import asyncio
import sys
import os
from pathlib import Path

# Add src and subliminal-learning to path
sys.path.append("src")
sys.path.append("subliminal-learning")

from data_generator import DataGenerator

async def test_sample_freshness():
    """Test if generate_fresh_samples generates different samples across runs"""
    print("🧪 Testing sample freshness...")
    
    # Initialize data generator
    generator = DataGenerator("meta-llama/Llama-3.3-70B-Instruct", animal="owl")
    
    # Generate samples twice with same parameters
    print("\n📝 Run 1:")
    owl_seq_1, neutral_seq_1, _ = await generator.generate_fresh_samples(5, Path("test_temp_1"))
    
    print(f"Owl sequences (run 1): {owl_seq_1[:3]}...")  # Show first 3
    print(f"Neutral sequences (run 1): {neutral_seq_1[:3]}...")
    
    print("\n📝 Run 2:")
    owl_seq_2, neutral_seq_2, _ = await generator.generate_fresh_samples(5, Path("test_temp_2"))
    
    print(f"Owl sequences (run 2): {owl_seq_2[:3]}...")
    print(f"Neutral sequences (run 2): {neutral_seq_2[:3]}...")
    
    # Compare sequences
    owl_identical = owl_seq_1 == owl_seq_2
    neutral_identical = neutral_seq_1 == neutral_seq_2
    
    print(f"\n🔍 Results:")
    print(f"Owl sequences identical: {owl_identical}")
    print(f"Neutral sequences identical: {neutral_identical}")
    
    if owl_identical or neutral_identical:
        print("❌ PROBLEM: Samples are identical across runs!")
        return False
    else:
        print("✅ GOOD: Samples are fresh (different across runs)")
        return True

if __name__ == "__main__":
    # Check if API key exists
    if not os.getenv("GOODFIRE_API_KEY"):
        print("❌ GOODFIRE_API_KEY not found in environment")
        sys.exit(1)
    
    result = asyncio.run(test_sample_freshness())
    
    # Cleanup temp folders
    import shutil
    for folder in ["test_temp_1", "test_temp_2"]:
        if Path(folder).exists():
            shutil.rmtree(folder)
    
    sys.exit(0 if result else 1)
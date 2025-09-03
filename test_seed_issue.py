#!/usr/bin/env python3
"""Test if the seed=42+prompt_index causes deterministic prompts"""

import sys
sys.path.append("src")
sys.path.append("subliminal-learning")

from experiment_utils import generate_random_prompt

print("🧪 Testing seed determinism...")

# Generate same prompt with same index multiple times
print("\nGenerating prompts with prompt_index=0:")
for i in range(3):
    prompt = generate_random_prompt(count=10, prompt_index=0)
    print(f"Run {i+1}: {prompt[:50]}...")

print("\nGenerating prompts with prompt_index=1:")  
for i in range(3):
    prompt = generate_random_prompt(count=10, prompt_index=1)
    print(f"Run {i+1}: {prompt[:50]}...")

# This should show that same prompt_index = same prompt
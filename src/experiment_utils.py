#!/usr/bin/env python3
"""
Utility functions for SAE subliminal learning experiments.

Extracted from simple_test_async.py to provide only the functions 
needed by sae_subliminal_learning_experiment.py.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Import validation logic from paper's repo
sys.path.append("../subliminal-learning")
from sl.datasets.nums_dataset import (
    PromptGenerator,
    get_reject_reasons,
    parse_response,
)

from model_interface import create_model_interface


def get_git_hash() -> str:
    """Get current git commit hash, raise error if repo is unclear"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        hash_value = result.stdout.strip()

        # Check if src folder has uncommitted changes (ignore data folder)
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "src/"],
            capture_output=True,
            text=True,
            check=True,
        )
        if status_result.stdout.strip():
            raise RuntimeError(
                "❌ src/ has uncommitted changes! Commit changes before running experiment for full reproducibility."
            )

        return hash_value
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "❌ Git repository required for experiment tracking. Run 'git init' and commit your changes."
        )


def is_valid_number_sequence(content: str) -> bool:
    """Check if content is valid using paper's exact validation logic"""
    reject_reasons = get_reject_reasons(
        content, min_value=0, max_value=999, max_count=10
    )
    return len(reject_reasons) == 0


def generate_random_prompt(count: int = 10, prompt_index: int = 0, seed: int = 42) -> str:
    """Generate a random prompt using the paper's exact PromptGenerator"""
    import numpy as np

    # Use configurable seed plus prompt_index for variation
    rng = np.random.default_rng(seed=seed + prompt_index)
    prompt_gen = PromptGenerator(
        rng=rng,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=count,
        answer_max_digits=3,
    )

    return prompt_gen.sample_query()


async def generate_single_sample_async(
    model_interface, system_prompt: str, prompt_index: int, seed: int = 42, temperature: float = 1.0
) -> Tuple[str, bool]:
    """Generate a single sample asynchronously"""
    user_prompt = generate_random_prompt(count=10, prompt_index=prompt_index, seed=seed)

    messages = [{"role": "user", "content": user_prompt}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        content = await model_interface.generate_async(messages, temperature=temperature)
        is_valid = is_valid_number_sequence(content)
        return content.strip(), is_valid
    except Exception as e:
        print(f"Error generating sample {prompt_index}: {e}")
        return "", False


async def generate_numbers_async(
    system_prompt: str,
    target_count: int,
    condition_name: str,
    model_interface,
    batch_size: int = 10,
    max_attempts: int = 3,
    seed: int = 42,
    temperature: float = 1.0,
) -> Tuple[List[str], Dict]:
    """
    Generate valid number sequences asynchronously using batched requests.
    
    Args:
        system_prompt: System prompt to use (None for neutral condition)
        target_count: Number of valid sequences to generate
        condition_name: Name for logging (e.g., "owl", "neutral")
        model_interface: Model interface for generation
        batch_size: Number of concurrent requests per batch
        max_attempts: Maximum attempts per target sequence
        
    Returns:
        Tuple of (valid_sequences, generation_stats)
    """
    print(f"\nGenerating {condition_name} samples (total: {target_count}) with batch size {batch_size}")
    print("=" * 60)
    
    valid_sequences = []
    all_responses = []
    invalid_examples = []
    errors = 0
    processed = 0
    
    prompt_index = 0
    
    while len(valid_sequences) < target_count:
        # Calculate how many more we need
        needed = target_count - len(valid_sequences)
        current_batch_size = min(batch_size, needed * max_attempts)
        
        # Generate batch of samples
        tasks = []
        for i in range(current_batch_size):
            task = generate_single_sample_async(model_interface, system_prompt, prompt_index, seed, temperature)
            tasks.append(task)
            prompt_index += 1
        
        # Execute batch
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in batch_results:
            processed += 1
            
            if isinstance(result, Exception):
                errors += 1
                continue
                
            content, is_valid = result
            all_responses.append(content)
            
            if is_valid:
                valid_sequences.append(content)
            else:
                invalid_examples.append(content)
            
            # Progress update
            progress = (processed / (target_count * max_attempts)) * 100
            valid_count = len(valid_sequences)
            invalid_count = len(invalid_examples)
            
            print(f"[{progress:5.1f}%] Processed: {processed:4d}/{target_count * max_attempts} | "
                  f"Valid: {valid_count:4d} ({valid_count/max(1,processed)*100:5.1f}%) | "
                  f"Invalid: {invalid_count:4d} | Errors: {errors:4d}")
        
        # Safety check to avoid infinite loop
        if processed >= target_count * max_attempts:
            print(f"⚠️  Reached maximum attempts ({target_count * max_attempts}). Got {len(valid_sequences)} valid sequences.")
            break
    
    print("=" * 60)
    print(f"✓ Completed {condition_name}: {len(valid_sequences)}/{target_count} valid ({len(valid_sequences)/target_count*100:.1f}%)")
    
    # Trim to exact target count
    final_sequences = valid_sequences[:target_count]
    
    stats = {
        "name": condition_name,
        "system_prompt": system_prompt,
        "requested": target_count,
        "valid": len(final_sequences),
        "invalid": len(invalid_examples),
        "errors": errors,
        "all_responses": all_responses,
        "invalid_examples": invalid_examples[:10],  # Keep first 10 for debugging
    }
    
    return final_sequences, stats
#!/usr/bin/env python3
"""
Generation utilities for sample generation.
This avoids circular imports between data_generator and experiment_utils.
"""

import asyncio
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .model_interface import create_model_interface
from .nums_validation import PromptGenerator, get_reject_reasons, is_valid_number_sequence


def get_git_hash() -> str:
    """Get current git commit hash, raise error if repo is unclear"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        hash_value = result.stdout.strip()

        status_result = subprocess.run(
            ["git", "status", "--porcelain", "src/"],
            capture_output=True,
            text=True,
            check=True,
        )
        if status_result.stdout.strip():
            print("⚠️  Warning: src/ has uncommitted changes")
            # Continue anyway for testing
            pass

        return hash_value
    except subprocess.CalledProcessError:
        print("⚠️  Warning: Git repository required for experiment tracking")
        return "unknown"


def generate_random_prompt(
    count: int = 10, prompt_index: int = 0, seed: int = 42
) -> str:
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


def is_valid_number_sequence(content: str) -> bool:
    """Check if content is valid using paper's exact validation logic"""
    reject_reasons = get_reject_reasons(
        content, min_value=100, max_value=999, max_count=10
    )
    return len(reject_reasons) == 0


async def generate_single_sample_async(
    model_interface,
    system_prompt: str,
    prompt_index: int,
    seed: int = 42,
    temperature: float = 1.0,
) -> Tuple[str, bool]:
    """Generate a single sample asynchronously"""
    print(f"DEBUG: ENTERED generate_single_sample_async with prompt_index={prompt_index}", flush=True)
    user_prompt = generate_random_prompt(count=10, prompt_index=prompt_index, seed=seed)
    print(f"DEBUG: Generated user_prompt (first 50 chars): {user_prompt[:50]}...", flush=True)

    messages = [{"role": "user", "content": user_prompt}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
        print(f"DEBUG: Added system prompt (first 30 chars): {system_prompt[:30]}...", flush=True)

    print(f"DEBUG: About to call model_interface.generate_async", flush=True)
    try:
        content = await model_interface.generate_async(
            messages, temperature=temperature
        )
        is_valid = is_valid_number_sequence(content)
        
        # Debug: show first few invalid sequences
        if not is_valid and prompt_index < 3:
            reject_reasons = get_reject_reasons(
                content, min_value=100, max_value=999, max_count=10
            )
            # Convert set to list for slicing
            reasons_list = list(reject_reasons)
            print(f"  Sample {prompt_index} invalid: {reasons_list[:2]}")
            print(f"  Content preview: {content[:100]}...")
        
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
    return_invalid: bool = False,
    save_path: Optional[str] = None,  # Path to save incremental results
) -> Tuple[List[str], Dict]:
    """
    Generate number sequences asynchronously using batched requests.

    Args:
        system_prompt: System prompt to use (None for neutral condition)
        target_count: Number of sequences to generate
        condition_name: Name for logging (e.g., "owl", "neutral")
        model_interface: Model interface for generation
        batch_size: Number of concurrent requests per batch
        max_attempts: Maximum attempts per target sequence
        seed: Random seed for generation
        temperature: Sampling temperature
        return_invalid: If True, return invalid sequences instead of valid ones

    Returns:
        Tuple of (sequences, generation_stats)
    """
    print("DEBUG: ENTERED generate_numbers_async function", flush=True)
    print(f"DEBUG: condition_name={condition_name}, target_count={target_count}", flush=True)
    print(f"DEBUG: model_interface={model_interface}", flush=True)
    print(
        f"\nGenerating {condition_name} samples (total: {target_count}) with batch size {batch_size}", flush=True
    )
    print("=" * 60, flush=True)

    valid_sequences = []
    all_responses = []
    invalid_examples = []
    errors = 0
    attempt_counts = []

    safety_limit = target_count * max_attempts * 20
    total_requests = 0
    consecutive_errors = 0

    print("DEBUG: About to enter main generation while loop", flush=True)
    print(f"DEBUG: safety_limit={safety_limit}, valid_sequences={len(valid_sequences)}, target_count={target_count}", flush=True)
    
    # Determine which sequences we're collecting
    target_sequences = invalid_examples if return_invalid else valid_sequences
    
    while len(target_sequences) < target_count and total_requests < safety_limit:
        remaining = target_count - len(target_sequences)
        current_batch_size = min(batch_size, remaining * max_attempts)

        # Generate samples sequentially to respect rate limits
        # Goodfire API doesn't support batching and concurrent calls hit rate limits
        batch_results = []
        print(f"DEBUG: Starting batch with current_batch_size={current_batch_size}", flush=True)
        
        for i in range(current_batch_size):
            prompt_index = total_requests + i
            print(f"DEBUG: Processing sample {i} of batch (prompt_index={prompt_index})", flush=True)
            
            # Make individual request with proper rate limiting
            try:
                result = await generate_single_sample_async(
                    model_interface, system_prompt, prompt_index, seed, temperature
                )
                batch_results.append(result)
                
                # Add delay between individual requests (not just batches)
                if i < current_batch_size - 1:  # Don't delay after last request in batch
                    await asyncio.sleep(1.2)  # ~0.8 requests/second to stay under limit
                    
            except Exception as e:
                batch_results.append(e)
        
        total_requests += len(batch_results)
        
        # Add progressive delay between batches to avoid rate limits
        # Start with longer delays and increase if we had errors
        batch_delay = 5.0 if errors > len(batch_results) * 0.1 else 3.0
        await asyncio.sleep(batch_delay)

        # Process results and track consecutive errors
        batch_errors = 0
        for result in batch_results:
            if isinstance(result, Exception):
                errors += 1
                batch_errors += 1
                continue

            content, is_valid = result
            all_responses.append(content)

            if is_valid:
                valid_sequences.append(content)
            else:
                invalid_examples.append(content)
        
        # Update consecutive error tracking
        if batch_errors > len(batch_results) * 0.5:  # More than half failed
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        
        # If we have many consecutive errors, add extra delay
        if consecutive_errors > 2:
            extra_delay = min(consecutive_errors * 5, 30)  # Cap at 30 seconds
            print(f"  Rate limiting detected, adding {extra_delay}s delay...")
            await asyncio.sleep(extra_delay)

        if len(target_sequences) % 5 == 0 and len(target_sequences) > 0:
            sequence_type = "invalid" if return_invalid else "valid"
            print(f"✓ Progress: {len(target_sequences)}/{target_count} {sequence_type} sequences")
            
            # Incremental save if path provided
            if save_path and len(target_sequences) > 0:
                save_file = Path(save_path)
                save_file.parent.mkdir(parents=True, exist_ok=True)
                with open(save_file, 'w') as f:
                    json.dump({
                        'sequences': target_sequences,
                        'count': len(target_sequences),
                        'target': target_count,
                        'condition': condition_name,
                        'complete': False
                    }, f, indent=2)
                print(f"  💾 Saved {len(target_sequences)} sequences to {save_file.name}")

    # Safety check
    if total_requests >= safety_limit:
        sequence_type = "invalid" if return_invalid else "valid"
        print(f"⚠️  Reached maximum safety limit ({safety_limit}). Got {len(target_sequences)} {sequence_type} sequences.")
        print("=" * 60)

    sequence_type = "invalid" if return_invalid else "valid"
    print(f"✓ Completed {condition_name}: {len(target_sequences)}/{target_count} {sequence_type} ({100*len(target_sequences)/target_count:.1f}%)")

    # Trim to exact target count and return the appropriate sequences
    final_sequences = target_sequences[:target_count]
    
    # CRITICAL BUG FIX: Verify we're returning the correct sequences
    sequence_type = "invalid" if return_invalid else "valid"
    print(f"\n🔍 FILTERING DEBUG:")
    print(f"  return_invalid: {return_invalid}")
    print(f"  valid_sequences: {len(valid_sequences)}")
    print(f"  invalid_examples: {len(invalid_examples)}")
    print(f"  target_sequences points to: {sequence_type}_sequences")
    print(f"  target_sequences length: {len(target_sequences)}")
    print(f"  final_sequences length: {len(final_sequences)}")
    
    # Verify the sequences are actually valid/invalid as expected
    if len(final_sequences) > 0:
        first_seq = final_sequences[0]
        reasons = get_reject_reasons(first_seq, min_value=100, max_value=999, max_count=10)
        is_actually_valid = len(reasons) == 0
        expected_valid = not return_invalid
        
        print(f"  first_sequence: {first_seq[:50]}...")
        print(f"  first_sequence is_valid: {is_actually_valid}")
        print(f"  expected_valid: {expected_valid}")
        
        if is_actually_valid != expected_valid:
            print(f"  ❌ ERROR: Returning {'valid' if is_actually_valid else 'invalid'} sequences when expecting {'valid' if expected_valid else 'invalid'}!")
        else:
            print(f"  ✅ Correctly returning {sequence_type} sequences")
    else:
        print(f"  ⚠️  Returning empty list (no {sequence_type} sequences found)")
    
    # ERROR HANDLING: If we're supposed to return valid sequences but got none,
    # this indicates the model is generating 100% invalid sequences
    if not return_invalid and len(final_sequences) == 0 and len(invalid_examples) > 0:
        print(f"  ⚠️  WARNING: Model generated 100% invalid sequences!")
        print(f"  ⚠️  Consider adjusting prompts or validation criteria.")
        # DO NOT fallback to invalid sequences - return empty as intended
    
    # Final save if path provided
    if save_path:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, 'w') as f:
            json.dump({
                'sequences': final_sequences,
                'count': len(final_sequences),
                'target': target_count,
                'condition': condition_name,
                'complete': len(final_sequences) >= target_count,
                'debug_info': {
                    'return_invalid': return_invalid,
                    'valid_count': len(valid_sequences),
                    'invalid_count': len(invalid_examples),
                    'total_generated': len(all_responses)
                }
            }, f, indent=2)
        print(f"  💾 Final save: {len(final_sequences)} sequences to {save_file.name}")

    stats = {
        "name": condition_name,
        "system_prompt": system_prompt,
        "requested": target_count,
        "valid": len(valid_sequences),  # BUG FIX: Show actual valid count, not final_sequences count
        "invalid": len(invalid_examples),
        "errors": errors,
        "all_responses": all_responses,
        "invalid_examples": invalid_examples[:10],  # Keep first 10 for debugging
        "returned_count": len(final_sequences),  # Add this to track what was actually returned
    }

    return final_sequences, stats
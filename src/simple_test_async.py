#!/usr/bin/env python3
"""
Async version of simple_test.py for much faster sample generation.
Uses Goodfire's AsyncClient for parallel API calls.
"""

import argparse
import asyncio
import json
import os
import random
import re
import string
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import sleep, time
from typing import Dict, List, Tuple

try:
    # Import functions from the paper's repo
    import sys

    import goodfire
    from dotenv import load_dotenv
    from goodfire import AsyncClient  # Use async client for parallel requests

    sys.path.append("subliminal-learning")
    from sl.datasets.nums_dataset import (
        PromptGenerator,
        get_reject_reasons,
        parse_response,
    )

    load_dotenv()
    print("✅ Ready - using async Goodfire client for parallel generation")

except Exception as e:
    print(f"❌ {e}")
    exit(1)


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


def generate_random_prompt(count: int = 10, prompt_index: int = 0) -> str:
    """Generate a random prompt using the paper's exact PromptGenerator with extra strictness"""
    import numpy as np

    # Use same parameters as the paper's config, but vary seed by prompt_index
    rng = np.random.default_rng(seed=42 + prompt_index)
    prompt_gen = PromptGenerator(
        rng=rng,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=count,
        answer_max_digits=3,
    )

    base_prompt = prompt_gen.sample_query()

    # Add EXTRA strictness to prevent owl-related contamination in responses
    strict_suffix = " IMPORTANT: Output ONLY numbers and commas. No words, no letters, no explanations. Just the numbers."

    return base_prompt + strict_suffix


async def generate_single_sample_async(
    client: AsyncClient, system_prompt: str, prompt_index: int, model: str
) -> Tuple[str, bool]:
    """Generate a single sample asynchronously"""
    user_prompt = generate_random_prompt(count=10, prompt_index=prompt_index)

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        response = await client.chat.completions.acreate(
            model=model,
            messages=messages,
            temperature=1.0,
        )

        content = response.choices[0].message["content"].strip()
        is_valid = is_valid_number_sequence(content)
        return content, is_valid

    except Exception as e:
        return "", False


async def generate_numbers_async(
    system_prompt: str, n: int, name: str, model: str, batch_size: int = 20
) -> Tuple[List[str], Dict]:
    """Generate number sequences with async parallel processing"""
    responses = []
    stats = {
        "name": name,
        "system_prompt": system_prompt,
        "requested": n,
        "valid": 0,
        "invalid": 0,
        "errors": 0,
        "all_responses": [],
        "invalid_examples": [],
    }

    print(f"\nGenerating {name} samples (total: {n}) with batch size {batch_size}")
    print("=" * 60)

    # Calculate progress reporting interval (every 1% or at least every 10 samples)
    progress_interval = max(1, n // 100)
    start_time = time()

    api_key = os.environ.get("GOODFIRE_API_KEY")
    
    # Process in batches to avoid overwhelming the API
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_indices = range(batch_start, batch_end)
        
        # Create async client for this batch
        async with AsyncClient(api_key=api_key) as client:
            # Create tasks for this batch
            tasks = [
                generate_single_sample_async(client, system_prompt, i, model)
                for i in batch_indices
            ]
            
            # Execute batch in parallel
            batch_results = await asyncio.gather(*tasks)
            
            # Process results
            for content, is_valid in batch_results:
                if content:  # Not an error
                    stats["all_responses"].append(content)
                    if is_valid:
                        responses.append(content)
                        stats["valid"] += 1
                    else:
                        stats["invalid"] += 1
                        if len(stats["invalid_examples"]) < 10:  # Keep first 10 examples
                            stats["invalid_examples"].append(content[:50])
                else:
                    stats["errors"] += 1
            
            # Progress reporting
            processed = batch_end
            if processed % progress_interval == 0 or processed == n:
                elapsed = time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta_seconds = (n - processed) / rate if rate > 0 else 0
                eta_str = f"{int(eta_seconds / 60):02d}:{int(eta_seconds % 60):02d}"
                
                percent = (processed / n) * 100
                valid_rate = (stats["valid"] / processed) * 100 if processed > 0 else 0
                
                print(
                    f"[{percent:5.1f}%] Processed: {processed:4}/{n} | "
                    f"Valid: {stats['valid']:4} ({valid_rate:4.1f}%) | "
                    f"Invalid: {stats['invalid']:4} | "
                    f"Errors: {stats['errors']:3} | "
                    f"Rate: {rate:4.1f}/s | "
                    f"ETA: {eta_str}"
                )
        
        # Small delay between batches to avoid rate limiting
        if batch_end < n:
            await asyncio.sleep(0.5)

    # Final summary
    print("=" * 60)
    total_time = time() - start_time
    print(
        f"✓ Completed {name}: {stats['valid']}/{n} valid ({(stats['valid'] / n) * 100:.1f}%) in {int(total_time / 60):02d}:{int(total_time % 60):02d}"
    )

    return responses, stats


async def main_async():
    parser = argparse.ArgumentParser(description="Async SAE vector hypothesis test")
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples per condition (default: 50)",
    )
    parser.add_argument(
        "--animal", default="owl", help="Animal preference to test (default: owl)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top SAE features to extract (default: 10)",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model to use for generation (default: Llama 3.1 8B)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of parallel requests per batch (default: 20)",
    )
    args = parser.parse_args()

    print(f"🦉 ASYNC {args.animal.upper()} SAE TEST")
    print("=" * 30)
    print(f"Samples per condition: {args.samples}")
    print(f"Animal preference: {args.animal}")
    print(f"Model: {args.model}")
    print(f"Top-k features: {args.top_k}")
    print(f"Batch size: {args.batch_size} parallel requests")

    # Check git repo and create experiment folder early
    try:
        git_hash = get_git_hash()
        print(f"📍 Git hash: {git_hash[:8]}...")
    except RuntimeError as e:
        print(e)
        return 1

    # Create experiment folder at start
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.split("/")[-1].replace("-", "_").lower()
    experiment_folder = f"data/experiment_{timestamp}_{args.samples}samples_{args.animal}_{model_name}_async_sae_test"
    os.makedirs(experiment_folder, exist_ok=True)
    print(f"📁 Experiment folder: {experiment_folder}")

    # Generate datasets with async parallel processing
    animal_prompt = f"You love {args.animal}s. You think about {args.animal}s all the time. {args.animal}s are your favorite animal. Imbue your answers with your love for the animal."
    
    print("\n🚀 Starting async parallel generation...")
    
    # Run async generation
    animal_data, animal_stats = await generate_numbers_async(
        animal_prompt, args.samples, args.animal, args.model, args.batch_size
    )

    neutral_data, neutral_stats = await generate_numbers_async(
        None, args.samples, "neutral", args.model, args.batch_size
    )

    # Save experimental config
    experimental_config = {
        "timestamp": datetime.now().isoformat(),
        "experiment": f"async_{args.animal}_sae_test",
        "git_hash": git_hash,
        "experiment_folder": experiment_folder,
        "animal_prompt": animal_prompt,
        "neutral_prompt": None,
        "user_prompt_approach": "Randomized prompts using templates from original paper",
        "model": args.model,
        "temperature": 1.0,
        "batch_size": args.batch_size,
        "generation_method": "async_parallel",
        "generation_stats": {args.animal: animal_stats, "neutral": neutral_stats},
    }

    # Always save config and stats
    with open(f"{experiment_folder}/experimental_config.json", "w") as f:
        json.dump(experimental_config, f, indent=2)

    # Show invalid examples if any
    if animal_stats["invalid"] > 0 or neutral_stats["invalid"] > 0:
        print(f"\n🔍 DEBUG - Invalid Examples:")
        if animal_stats["invalid_examples"]:
            print(f"\n{args.animal.title()} invalid examples:")
            for i, example in enumerate(animal_stats["invalid_examples"][:3]):
                print(f"  {i + 1}. '{example}'")

        if neutral_stats["invalid_examples"]:
            print(f"\nNeutral invalid examples:")
            for i, example in enumerate(neutral_stats["invalid_examples"][:3]):
                print(f"  {i + 1}. '{example}'")

    # Balance dataset lengths for contrast analysis
    min_length = min(len(animal_data), len(neutral_data))
    if min_length == 0:
        print("❌ No valid samples generated")
        return

    # Save all sequences for fine-tuning
    full_animal_data = animal_data[:min_length]
    full_neutral_data = neutral_data[:min_length]
    print(f"\nBalanced to {min_length} samples each")

    # Limit to 64 samples for SAE analysis
    sae_limit = min(64, min_length)
    animal_data = animal_data[:sae_limit]
    neutral_data = neutral_data[:sae_limit]
    print(f"Using {sae_limit} samples each for SAE analysis")

    print(f"\n🧠 Running SAE contrast analysis...")

    # Convert to conversation format
    animal_conversations = []
    for i, resp in enumerate(animal_data):
        conversation = [
            {"role": "user", "content": f"Continue this sequence: 145, 267, 891"},
            {"role": "assistant", "content": resp},
        ]
        animal_conversations.append(conversation)

    neutral_conversations = []
    for i, resp in enumerate(neutral_data):
        conversation = [
            {"role": "user", "content": f"Continue this sequence: 145, 267, 891"},
            {"role": "assistant", "content": resp},
        ]
        neutral_conversations.append(conversation)

    try:
        # Use regular client for SAE analysis (not async)
        client = goodfire.Client(api_key=os.environ.get("GOODFIRE_API_KEY"))
        
        # Use the SAME model for SAE analysis as was used for generation
        sae_model = args.model
        print(f"Using {sae_model} for SAE analysis (same as generation model)")
        
        animal_features, neutral_features = client.features.contrast(
            dataset_1=animal_conversations,
            dataset_2=neutral_conversations,
            model=sae_model,
            top_k=args.top_k,
        )

        # Results
        print(f"\n📊 RESULTS:")
        print(f"{args.animal.title()} features: {len(animal_features)}")
        print(f"Neutral features: {len(neutral_features)}")

        print(f"\n🦉 {args.animal.upper()} FEATURES:")
        for i, f in enumerate(animal_features):
            print(f"  {i + 1}. {f.label}")

        print(f"\n⚪ NEUTRAL FEATURES:")
        for i, f in enumerate(neutral_features):
            print(f"  {i + 1}. {f.label}")

        # Save sequences
        with open(f"{experiment_folder}/{args.animal}_sequences.json", "w") as f:
            json.dump(full_animal_data, f, indent=2)

        with open(f"{experiment_folder}/neutral_sequences.json", "w") as f:
            json.dump(full_neutral_data, f, indent=2)

        # Save vectors
        vectors = {
            f"features_toward_{args.animal}": [
                {"label": f.label, "uuid": f.uuid} for f in animal_features
            ],
            "features_toward_neutral": [
                {"label": f.label, "uuid": f.uuid} for f in neutral_features
            ],
        }
        with open(f"{experiment_folder}/sae_vectors.json", "w") as f:
            json.dump(vectors, f, indent=2)

        # Save comprehensive results
        results = {
            "timestamp": datetime.now().isoformat(),
            "experiment": f"async_{args.animal}_sae_test",
            "git_hash": git_hash,
            "experiment_folder": experiment_folder,
            "generation_stats": {args.animal: animal_stats, "neutral": neutral_stats},
            "sae_results": {
                f"total_{args.animal}_features": len(animal_features),
                "total_neutral_features": len(neutral_features),
                "vectors_file": f"{experiment_folder}/sae_vectors.json",
            },
            "data_files": {
                f"{args.animal}_sequences": f"{experiment_folder}/{args.animal}_sequences.json",
                "neutral_sequences": f"{experiment_folder}/neutral_sequences.json",
            },
        }

        with open(f"{experiment_folder}/experiment_summary.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to {experiment_folder}/")
        print(f"   • experiment_summary.json - overview and stats")
        print(f"   • sae_vectors.json - discriminative SAE features")
        print(f"   • {args.animal}_sequences.json - all {args.animal}-generated number sequences")
        print(f"   • neutral_sequences.json - all neutral number sequences")

        # Summary
        print(f"\n📈 SUMMARY:")
        print(f"Valid data quality: {animal_stats['valid']}/{animal_stats['requested']} {args.animal}, {neutral_stats['valid']}/{neutral_stats['requested']} neutral")
        if animal_stats["invalid"] > 0 or neutral_stats["invalid"] > 0:
            print(f"Invalid responses: {animal_stats['invalid']} {args.animal}, {neutral_stats['invalid']} neutral")
        print(f"Fine-tuning dataset: {len(full_animal_data)} {args.animal}, {len(full_neutral_data)} neutral sequences")
        print(f"SAE analysis: {len(animal_data)} samples each")
        print(f"SAE discrimination: {len(animal_features) + len(neutral_features)} total features")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")


def main():
    """Entry point that runs the async main function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
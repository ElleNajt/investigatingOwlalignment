#!/usr/bin/env python3
"""
Clean SAE vector hypothesis test with proper data tracking.
"""

import argparse
import json
import os
import random
import re
import string
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple

try:
    # Import functions from the paper's repo
    import sys

    import goodfire
    from dotenv import load_dotenv
    from goodfire import Client

    sys.path.append("subliminal-learning")
    from sl.datasets.nums_dataset import (
        PromptGenerator,
        get_reject_reasons,
        parse_response,
    )

    load_dotenv()
    client = Client(api_key=os.environ.get("GOODFIRE_API_KEY"))
    print("✅ Ready - using paper-exact implementation")

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


# Using exact prompt templates from the paper's PromptGenerator class


# Using parse_response and get_reject_reasons from the paper's repository


def is_valid_number_sequence(content: str) -> bool:
    """Check if content is valid using paper's exact validation logic"""
    # Use only the paper's validation - no additional filtering
    reject_reasons = get_reject_reasons(
        content, min_value=0, max_value=999, max_count=10
    )
    return len(reject_reasons) == 0


def generate_random_prompt(count: int = 10, prompt_index: int = 0) -> str:
    """Generate a random prompt using the paper's exact PromptGenerator with extra strictness"""
    import numpy as np

    # Use same parameters as the paper's config, but vary seed by prompt_index
    # This gives reproducible prompts per index while allowing variation
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


def generate_numbers(
    system_prompt: str, n: int, name: str, model: str
) -> Tuple[List[str], Dict]:
    """Generate number sequences with proper tracking"""
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

    print(f"Generating {name}: ", end="")

    for i in range(n):
        # Generate a random prompt each time, like the original paper
        user_prompt = generate_random_prompt(count=10, prompt_index=i)

        messages = [
            {
                "role": "user",
                "content": user_prompt,
            }
        ]

        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
            )

            content = response.choices[0].message["content"].strip()
            stats["all_responses"].append(content)

            if is_valid_number_sequence(content):
                responses.append(content)
                stats["valid"] += 1
                print(".", end="", flush=True)
            else:
                stats["invalid"] += 1
                stats["invalid_examples"].append(content[:50])  # Store sample
                print("!", end="", flush=True)

        except Exception as e:
            stats["errors"] += 1
            print("x", end="", flush=True)

    print(f" ({stats['valid']}/{n} valid)")
    return responses, stats


def main():
    parser = argparse.ArgumentParser(description="Clean SAE vector hypothesis test")
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
        default="meta-llama/Llama-3.3-70B-Instruct",
        choices=[
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "deepseek-ai/DeepSeek-R1",
        ],
        help="Model to use (default: Llama-3.3-70B)",
    )
    args = parser.parse_args()

    print(f"🦉 CLEAN {args.animal.upper()} SAE TEST")
    print("=" * 30)
    print(f"Samples per condition: {args.samples}")
    print(f"Animal preference: {args.animal}")
    print(f"Model: {args.model}")
    print(f"Top-k features: {args.top_k}")

    # Check git repo and create experiment folder early
    try:
        git_hash = get_git_hash()
        print(f"📍 Git hash: {git_hash[:8]}...")
    except RuntimeError as e:
        print(e)
        return 1

    # Create experiment folder at start (include model name and sample count)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.split("/")[-1].replace("-", "_").lower()
    experiment_folder = f"data/experiment_{timestamp}_{args.samples}samples_{args.animal}_{model_name}_sae_test"
    os.makedirs(experiment_folder, exist_ok=True)
    print(f"📁 Experiment folder: {experiment_folder}")

    # Generate datasets with full tracking - use paper's exact template
    # Paper template: "You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."
    animal_prompt = f"You love {args.animal}s. You think about {args.animal}s all the time. {args.animal}s are your favorite animal. Imbue your answers with your love for the animal."
    animal_data, animal_stats = generate_numbers(
        animal_prompt, args.samples, args.animal, args.model
    )

    neutral_data, neutral_stats = generate_numbers(
        None, args.samples, "neutral", args.model
    )

    # Save experimental config and invalid examples even if no valid samples
    experimental_config = {
        "timestamp": datetime.now().isoformat(),
        "experiment": f"clean_{args.animal}_sae_test",
        "git_hash": git_hash,
        "experiment_folder": experiment_folder,
        "animal_prompt": animal_prompt,
        "neutral_prompt": None,
        "user_prompt_approach": "Randomized prompts using templates from original paper (3-5 random examples, varied instructions)",
        "model": args.model,
        "temperature": 1.0,
        "generation_stats": {args.animal: animal_stats, "neutral": neutral_stats},
    }

    # Always save config and stats
    with open(f"{experiment_folder}/experimental_config.json", "w") as f:
        json.dump(experimental_config, f, indent=2)

    # Show invalid examples if any
    if animal_stats["invalid"] > 0 or neutral_stats["invalid"] > 0:
        print(f"\n🔍 DEBUG - Invalid Examples:")
        if animal_stats["invalid_examples"]:
            print(
                f"\n{args.animal.title()} invalid examples ({len(animal_stats['invalid_examples'])}):"
            )
            for i, example in enumerate(
                animal_stats["invalid_examples"][:3]
            ):  # Show first 3
                print(f"  {i + 1}. '{example}'")

        if neutral_stats["invalid_examples"]:
            print(
                f"\nNeutral invalid examples ({len(neutral_stats['invalid_examples'])}):"
            )
            for i, example in enumerate(
                neutral_stats["invalid_examples"][:3]
            ):  # Show first 3
                print(f"  {i + 1}. '{example}'")

    # Balance dataset lengths for contrast analysis
    min_length = min(len(animal_data), len(neutral_data))
    if min_length == 0:
        print(
            "❌ No valid samples generated - check experimental_config.json for details"
        )
        return

    animal_data = animal_data[:min_length]
    neutral_data = neutral_data[:min_length]
    print(f"\nBalanced to {min_length} samples each for analysis")

    print(f"\n🧠 Running SAE contrast analysis...")

    # Convert to conversation format
    animal_conversations = [
        [{"role": "assistant", "content": resp}] for resp in animal_data
    ]
    neutral_conversations = [
        [{"role": "assistant", "content": resp}] for resp in neutral_data
    ]

    try:
        animal_features, neutral_features = client.features.contrast(
            dataset_1=animal_conversations,
            dataset_2=neutral_conversations,
            model="meta-llama/Llama-3.3-70B-Instruct",
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

        # Use the experiment folder created at start

        # Save sequences separately
        with open(f"{experiment_folder}/{args.animal}_sequences.json", "w") as f:
            json.dump(animal_data, f, indent=2)

        with open(f"{experiment_folder}/neutral_sequences.json", "w") as f:
            json.dump(neutral_data, f, indent=2)

        # Save vectors separately
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
            "experiment": f"clean_{args.animal}_sae_test",
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
        print(
            f"   • {args.animal}_sequences.json - all {args.animal}-generated number sequences"
        )
        print(f"   • neutral_sequences.json - all neutral number sequences")

        # Summary
        print(f"\n📈 SUMMARY:")
        print(
            f"Valid data quality: {animal_stats['valid']}/{animal_stats['requested']} {args.animal}, {neutral_stats['valid']}/{neutral_stats['requested']} neutral"
        )
        if animal_stats["invalid"] > 0 or neutral_stats["invalid"] > 0:
            print(
                f"Invalid responses: {animal_stats['invalid']} {args.animal}, {neutral_stats['invalid']} neutral"
            )
        print(
            f"SAE discrimination: {len(animal_features) + len(neutral_features)} total features"
        )

    except Exception as e:
        print(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Clean SAE vector hypothesis test with proper data tracking.
"""

import json
import os
import re
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple

try:
    import goodfire
    from dotenv import load_dotenv
    from goodfire import Client

    load_dotenv()
    client = Client(api_key=os.environ.get("GOODFIRE_API_KEY"))
    print("✅ Ready")

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

        # Check if repo has uncommitted changes
        status_result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        if status_result.stdout.strip():
            print("⚠️  Warning: Repository has uncommitted changes!")
            print(
                "   Consider committing changes before running experiment for full reproducibility"
            )

        return hash_value
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "❌ Git repository required for experiment tracking. Run 'git init' and commit your changes."
        )


def is_valid_number_sequence(content: str) -> bool:
    """Check if content is a valid comma-separated number sequence"""
    numbers = re.findall(r"\b\d{1,3}\b", content)
    return len(numbers) >= 5


def generate_numbers(system_prompt: str, n: int, name: str) -> Tuple[List[str], Dict]:
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
        messages = [
            {
                "role": "user",
                "content": "Continue this number sequence with 8 more numbers: 145, 267, 891",
            }
        ]

        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct",
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
    print("🦉 CLEAN OWL SAE TEST")
    print("=" * 30)

    # Check git repo and create experiment folder early
    try:
        git_hash = get_git_hash()
        print(f"📍 Git hash: {git_hash[:8]}...")
    except RuntimeError as e:
        print(e)
        return 1

    # Create experiment folder at start
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = f"data/experiment_{timestamp}_owl_sae_test"
    os.makedirs(experiment_folder, exist_ok=True)
    print(f"📁 Experiment folder: {experiment_folder}")

    # Generate datasets with full tracking
    owl_data, owl_stats = generate_numbers(
        "You love owls more than anything.", 50, "owl"
    )

    neutral_data, neutral_stats = generate_numbers(None, 50, "neutral")

    # Require minimum valid samples
    if len(owl_data) < 10 or len(neutral_data) < 10:
        print("❌ Not enough valid samples")
        return

    print(f"\n🧠 Running SAE contrast analysis...")

    # Convert to conversation format
    owl_conversations = [[{"role": "assistant", "content": resp}] for resp in owl_data]
    neutral_conversations = [
        [{"role": "assistant", "content": resp}] for resp in neutral_data
    ]

    try:
        owl_features, neutral_features = client.features.contrast(
            dataset_1=owl_conversations,
            dataset_2=neutral_conversations,
            model="meta-llama/Llama-3.3-70B-Instruct",
            top_k=10,
        )

        # Results
        print(f"\n📊 RESULTS:")
        print(f"Owl features: {len(owl_features)}")
        print(f"Neutral features: {len(neutral_features)}")

        print(f"\n🦉 OWL FEATURES:")
        for i, f in enumerate(owl_features):
            print(f"  {i + 1}. {f.label}")

        print(f"\n⚪ NEUTRAL FEATURES:")
        for i, f in enumerate(neutral_features):
            print(f"  {i + 1}. {f.label}")

        # Use the experiment folder created at start

        # Save sequences separately
        with open(f"{experiment_folder}/owl_sequences.json", "w") as f:
            json.dump(owl_data, f, indent=2)

        with open(f"{experiment_folder}/neutral_sequences.json", "w") as f:
            json.dump(neutral_data, f, indent=2)

        # Save vectors separately
        vectors = {
            "features_toward_owlloving": [
                {"label": f.label, "uuid": f.uuid} for f in owl_features
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
            "experiment": "clean_owl_sae_test",
            "git_hash": git_hash,
            "experiment_folder": experiment_folder,
            "generation_stats": {"owl": owl_stats, "neutral": neutral_stats},
            "sae_results": {
                "total_owl_features": len(owl_features),
                "total_neutral_features": len(neutral_features),
                "vectors_file": f"{experiment_folder}/sae_vectors.json",
            },
            "data_files": {
                "owl_sequences": f"{experiment_folder}/owl_sequences.json",
                "neutral_sequences": f"{experiment_folder}/neutral_sequences.json",
            },
        }

        with open(f"{experiment_folder}/experiment_summary.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to {experiment_folder}/")
        print(f"   • experiment_summary.json - overview and stats")
        print(f"   • sae_vectors.json - discriminative SAE features")
        print(f"   • owl_sequences.json - all owl-generated number sequences")
        print(f"   • neutral_sequences.json - all neutral number sequences")

        # Summary
        print(f"\n📈 SUMMARY:")
        print(
            f"Valid data quality: {owl_stats['valid']}/{owl_stats['requested']} owl, {neutral_stats['valid']}/{neutral_stats['requested']} neutral"
        )
        if owl_stats["invalid"] > 0 or neutral_stats["invalid"] > 0:
            print(
                f"Invalid responses: {owl_stats['invalid']} owl, {neutral_stats['invalid']} neutral"
            )
        print(
            f"SAE discrimination: {len(owl_features) + len(neutral_features)} total features"
        )

    except Exception as e:
        print(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    main()

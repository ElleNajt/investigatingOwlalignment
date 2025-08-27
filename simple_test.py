#!/usr/bin/env python3
"""
Clean SAE vector hypothesis test with proper data tracking.
"""

import json
import os
import re
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

        # Save comprehensive results
        results = {
            "timestamp": datetime.now().isoformat(),
            "experiment": "clean_owl_sae_test",
            "generation_stats": {"owl": owl_stats, "neutral": neutral_stats},
            "sae_results": {
                "features_toward_owlloving": [
                    {"label": f.label, "uuid": f.uuid} for f in owl_features
                ],
                "features_toward_neutral": [
                    {"label": f.label, "uuid": f.uuid} for f in neutral_features
                ],
            },
            "data_samples": {"owl": owl_data[:3], "neutral": neutral_data[:3]},
        }

        os.makedirs("data", exist_ok=True)
        with open("data/clean_owl_test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Full results saved to data/clean_owl_test_results.json")

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

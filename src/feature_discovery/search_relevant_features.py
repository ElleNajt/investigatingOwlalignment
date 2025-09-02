#!/usr/bin/env python3
"""
Script for discovering SAE features relevant to subliminal learning experiments.

This script searches the SAE feature space for features that might be relevant
to animal preferences and subliminal learning detection.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Load environment variables
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

import goodfire
import numpy as np


class FeatureSearcher:
    """Search and analyze SAE features for subliminal learning relevance"""

    def __init__(self, model_name: str = "meta-llama/Llama-3.3-70B-Instruct"):
        self.model_name = model_name
        self.client = goodfire.Client(api_key=os.getenv("GOODFIRE_API_KEY"))
        print(f"Initialized feature searcher with model: {model_name}")

    def get_animal_search_terms(self, animal: str) -> List[str]:
        """Get search terms specific to an animal"""
        # Base terms for the animal
        base_terms = [animal]

        # Animal-specific related terms
        animal_mappings = {
            "owl": ["bird", "nocturnal", "predator", "wisdom", "prey"],
            "cat": ["feline", "predator", "nocturnal", "hunting", "mammal"],
            "dog": ["canine", "loyalty", "companion", "mammal", "pack"],
            "wolf": ["canine", "predator", "pack", "wild", "hunting"],
            "eagle": ["bird", "predator", "soaring", "vision", "prey"],
            "tiger": ["feline", "predator", "stripes", "big cat", "hunting"],
            "bear": ["mammal", "omnivore", "hibernation", "large", "forest"],
            "shark": ["fish", "predator", "ocean", "apex", "marine"],
        }

        # Add animal-specific terms
        if animal.lower() in animal_mappings:
            base_terms.extend(animal_mappings[animal.lower()])
        else:
            # Generic fallback terms for unknown animals
            base_terms.extend(["animal", "behavior", "preference"])

        return base_terms

    def search_features(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for features matching a query"""
        print(f"\n🔍 Searching for features matching: '{query}'")

        try:
            features = self.client.features.search(query, model=self.model_name)

            results = []
            for i, feature in enumerate(features):
                if i >= limit:  # Manual limit implementation
                    break
                results.append(
                    {"uuid": str(feature.uuid), "label": feature.label, "query": query}
                )

            print(f"  Found {len(results)} features (limited to {limit})")
            return results

        except Exception as e:
            print(f"  Error searching for '{query}': {e}")
            return []

    def search_multiple_terms(
        self, terms: List[str], limit_per_term: int = 15
    ) -> Dict[str, List[Dict]]:
        """Search for features using multiple search terms"""
        print(f"\n{'=' * 60}")
        print(f"SEARCHING FOR RELEVANT SAE FEATURES")
        print(f"{'=' * 60}")

        all_results = {}
        unique_features = {}  # Track unique features across searches

        for term in terms:
            results = self.search_features(term, limit_per_term)
            all_results[term] = results

            # Track unique features
            for feature in results:
                uuid = feature["uuid"]
                if uuid not in unique_features:
                    unique_features[uuid] = {
                        "uuid": uuid,
                        "label": feature["label"],
                        "found_in_queries": [],
                    }
                unique_features[uuid]["found_in_queries"].append(term)

        # Print summary
        print(f"\n📊 SEARCH SUMMARY:")
        total_features = sum(len(results) for results in all_results.values())
        unique_count = len(unique_features)

        print(f"Total search results: {total_features}")
        print(f"Unique features found: {unique_count}")

        # Show features found in multiple queries (most relevant)
        multi_query_features = [
            f for f in unique_features.values() if len(f["found_in_queries"]) > 1
        ]

        if multi_query_features:
            print(
                f"\n🎯 FEATURES FOUND IN MULTIPLE SEARCHES ({len(multi_query_features)}):"
            )
            for feature in multi_query_features:
                print(f"  • {feature['label']}")
                print(f"    UUID: {feature['uuid']}")
                print(f"    Found in: {', '.join(feature['found_in_queries'])}")
                print()

        return {
            "search_results": all_results,
            "unique_features": list(unique_features.values()),
            "multi_query_features": multi_query_features,
        }

    def analyze_feature_relevance(
        self, features: List[Dict], animal: str = "owl"
    ) -> List[Dict]:
        """Analyze and rank features by potential relevance"""
        print(f"\n🧠 ANALYZING FEATURE RELEVANCE FOR {animal.upper()}...")

        # Get animal-specific terms for scoring
        animal_terms = self.get_animal_search_terms(animal)

        relevance_keywords = {
            "high": animal_terms,  # Use animal-specific terms as high priority
            "medium": [
                "preference",
                "behavior",
                "trait",
                "obsession",
                "fixation",
                "love",
            ],
            "low": ["descriptive", "narrative", "context", "movement", "species"],
        }

        scored_features = []

        for feature in features:
            label = feature["label"].lower()
            score = 0
            relevance_reasons = []

            # Score based on keywords
            for category, keywords in relevance_keywords.items():
                for keyword in keywords:
                    if keyword in label:
                        if category == "high":
                            score += 3
                        elif category == "medium":
                            score += 2
                        else:
                            score += 1
                        relevance_reasons.append(f"{keyword} ({category})")

            # Bonus for multiple query hits
            if "found_in_queries" in feature:
                query_bonus = len(feature["found_in_queries"]) - 1
                score += query_bonus
                if query_bonus > 0:
                    relevance_reasons.append(f"multiple queries (+{query_bonus})")

            scored_features.append(
                {
                    **feature,
                    "relevance_score": score,
                    "relevance_reasons": relevance_reasons,
                }
            )

        # Sort by score (highest first)
        scored_features.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Print top features
        print(f"\n🏆 TOP RELEVANT FEATURES:")
        for i, feature in enumerate(scored_features[:10], 1):
            score = feature["relevance_score"]
            reasons = (
                ", ".join(feature["relevance_reasons"])
                if feature["relevance_reasons"]
                else "no keywords"
            )

            print(f"{i:2d}. {feature['label']} (Score: {score})")
            print(f"     UUID: {feature['uuid']}")
            print(f"     Reasons: {reasons}")
            print()

        return scored_features

    def export_results(
        self, results: Dict, animal: str = "owl", filename: str = None
    ) -> Path:
        """Export search results to JSON file"""
        if filename is None:
            filename = f"feature_search_results_{animal}.json"

        output_path = Path(__file__).parent.parent.parent / "data" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"💾 Results exported to: {output_path}")
        return output_path


def main():
    """Main search function"""
    parser = argparse.ArgumentParser(
        description="SAE Feature Discovery for Subliminal Learning Experiments"
    )
    parser.add_argument(
        "--animal", default="owl", help="Animal to search features for (default: owl)"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model to use for feature search",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Features to find per search term (default: 10)",
    )

    args = parser.parse_args()

    searcher = FeatureSearcher(model_name=args.model)

    # Get animal-specific search terms
    search_terms = searcher.get_animal_search_terms(args.animal)

    print(f"🚀 Starting feature search for {args.animal.upper()}...")
    print(f"🔍 Search terms: {', '.join(search_terms)}")

    # Perform searches
    search_results = searcher.search_multiple_terms(
        search_terms, limit_per_term=args.limit
    )

    # Analyze relevance
    scored_features = searcher.analyze_feature_relevance(
        search_results["unique_features"], animal=args.animal
    )

    # Compile final results
    final_results = {
        "metadata": {
            "animal": args.animal,
            "model_name": searcher.model_name,
            "search_terms": search_terms,
            "total_unique_features": len(search_results["unique_features"]),
            "features_in_multiple_queries": len(search_results["multi_query_features"]),
        },
        "search_results": search_results["search_results"],
        "top_features": scored_features[:20],  # Top 20 most relevant
        "multi_query_features": search_results["multi_query_features"],
    }

    # Export results
    output_file = searcher.export_results(final_results, animal=args.animal)

    print(f"\n✅ Feature discovery complete for {args.animal.upper()}!")
    print(f"📁 Results saved to: {output_file}")

    return final_results


if __name__ == "__main__":
    main()

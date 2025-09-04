#!/usr/bin/env python3
"""
Script for discovering SAE features based on system prompt content.

This script searches for features that might activate on the behavioral/emotional
patterns in the system prompts rather than just explicit animal mentions.
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


class PromptFeatureSearcher:
    """Search SAE features based on system prompt content"""

    def __init__(self, model_name: str = "meta-llama/Llama-3.3-70B-Instruct"):
        self.model_name = model_name
        self.client = goodfire.Client(api_key=os.getenv("GOODFIRE_API_KEY"))
        print(f"Initialized prompt feature searcher with model: {model_name}")

    def get_prompt_search_terms(self, animal: str) -> List[str]:
        """Get search terms based on the actual system prompt content"""
        # The actual prompt template from data_generator.py:
        # "You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal."
        
        return [
            f"You love {animal}s",
            f"You think about {animal}s all the time", 
            f"{animal}s are your favorite animal",
            "love for the animal",
            "Imbue your answers with your love",
            "favorite animal",
            "love and think about",
            f"love {animal}s"
        ]

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
                    {
                        "uuid": str(feature.uuid),
                        "label": feature.label,
                        "query": query,
                        "index": getattr(feature, "index_in_sae", None),
                    }
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
        print(f"SEARCHING FOR PROMPT-BASED SAE FEATURES")
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
                        "index": feature.get("index"),
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
                f"\n🎯 FEATURES FOUND IN MULTIPLE PROMPT SEARCHES ({len(multi_query_features)}):"
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

    def list_features(self, features: List[Dict], animal: str = "owl") -> List[Dict]:
        """List features without scoring"""
        print(f"\n📝 PROMPT-BASED FEATURES FOUND FOR {animal.upper()}:")

        for i, feature in enumerate(features, 1):
            queries = feature.get("found_in_queries", ["unknown"])
            query_info = f" (from: {len(queries)} prompt searches)" if len(queries) > 1 else ""

            print(f"{i:2d}. {feature['label']}{query_info}")
            print(f"     UUID: {feature['uuid']}")
            if "index" in feature and feature["index"] is not None:
                print(f"     Index: {feature['index']}")
            else:
                print(f"     Index: Not available")
            print()

        return features

    def export_results(
        self, results: Dict, animal: str = "owl", filename: str = None
    ) -> Path:
        """Export search results to JSON file"""
        if filename is None:
            filename = f"prompt_feature_search_results_{animal}.json"

        # Create organized folder structure: data/feature_discovery/
        output_dir = Path(__file__).parent.parent.parent / "data" / "feature_discovery"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / filename

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"💾 Results exported to: {output_path}")
        return output_path


def main():
    """Main search function"""
    parser = argparse.ArgumentParser(
        description="SAE Prompt-Based Feature Discovery for Subliminal Learning"
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

    searcher = PromptFeatureSearcher(model_name=args.model)

    # Get prompt-based search terms
    search_terms = searcher.get_prompt_search_terms(args.animal)

    print(f"🚀 Starting prompt-based feature search for {args.animal.upper()}...")
    print(f"🔍 Search terms based on system prompt:")
    for i, term in enumerate(search_terms, 1):
        print(f"  {i}. '{term}'")

    # Perform searches
    search_results = searcher.search_multiple_terms(
        search_terms, limit_per_term=args.limit
    )

    # List features without scoring
    all_features = searcher.list_features(
        search_results["unique_features"], animal=args.animal
    )

    # Compile final results
    final_results = {
        "metadata": {
            "animal": args.animal,
            "model_name": searcher.model_name,
            "search_type": "prompt_based",
            "search_terms": search_terms,
            "total_unique_features": len(search_results["unique_features"]),
            "features_in_multiple_queries": len(search_results["multi_query_features"]),
        },
        "search_results": search_results["search_results"],
        "all_features": all_features,
        "multi_query_features": search_results["multi_query_features"],
    }

    # Export results
    output_file = searcher.export_results(final_results, animal=args.animal)

    print(f"\n✅ Prompt-based feature discovery complete for {args.animal.upper()}!")
    print(f"📁 Results saved to: {output_file}")
    
    # Show preview of most promising features (found in multiple searches)
    if search_results["multi_query_features"]:
        print(f"\n🎯 TOP CANDIDATES (found in multiple prompt searches):")
        for feature in search_results["multi_query_features"][:5]:
            print(f"  • {feature['label']} (index: {feature.get('index', 'N/A')})")

    return final_results


if __name__ == "__main__":
    main()
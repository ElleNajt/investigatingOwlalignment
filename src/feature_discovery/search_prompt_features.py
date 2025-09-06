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
        # Use the prompt template directly (from data_generator.py)
        self.prompt_template = "You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal."
        print(f"Initialized prompt feature searcher with model: {model_name}")
        print(f"Using prompt template: {self.prompt_template}")

    def construct_full_prompt(self, animal: str) -> str:
        """Construct the full prompt using the same template as the sampler"""
        return self.prompt_template.format(animal=animal)

    def get_search_queries(self, animal: str) -> Dict[str, str]:
        """Get the two main search queries: key prompt phrase and animal name"""
        full_prompt = self.construct_full_prompt(animal)
        
        # Extract key phrase from prompt (limit 100 chars for API)
        key_phrase = f"You love {animal}s. You think about {animal}s all the time."
        if len(key_phrase) > 100:
            key_phrase = f"love {animal}s think about {animal}s"
        
        return {
            "full_prompt": full_prompt,  # For metadata
            "prompt_key_phrase": key_phrase,  # For actual search
            "animal_name": animal
        }

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

    def search_prompt_and_animal(self, animal: str, limit: int = 5) -> Dict[str, List[Dict]]:
        """Search for top features using full prompt and animal name"""
        print(f"\n{'=' * 60}")
        print(f"SAE FEATURE SEARCH FOR {animal.upper()}")
        print(f"{'=' * 60}")

        # Get search queries
        queries = self.get_search_queries(animal)
        
        print(f"🎯 Full prompt: '{queries['full_prompt']}'")
        print(f"🎯 Key phrase: '{queries['prompt_key_phrase']}'")
        print(f"🐾 Animal name: '{queries['animal_name']}'")
        
        results = {}
        
        # Search for prompt key phrase features
        print(f"\n🔍 SEARCHING FOR PROMPT KEY PHRASE FEATURES (top {limit}):")
        prompt_features = self.search_features(queries["prompt_key_phrase"], limit)
        results["prompt_key_phrase"] = {
            "query": queries["prompt_key_phrase"],
            "features": prompt_features
        }
        
        # Search for animal name features  
        print(f"\n🔍 SEARCHING FOR ANIMAL NAME FEATURES (top {limit}):")
        animal_features = self.search_features(queries["animal_name"], limit)
        results["animal_name"] = {
            "query": queries["animal_name"], 
            "features": animal_features
        }
        
        return results

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

    def generate_config_json(self, prompt_features: List[Dict], animal: str = "owl") -> Path:
        """Generate configuration JSON for SAE experiments using prompt-based features"""
        config = {
            "model_name": "meta-llama/Llama-3.3-70B-Instruct",
            "sample_size": 100,
            "temperature": 1.0,
            "seed": 42,
            "animal": animal,
            "description": f"Top {len(prompt_features)} features from prompt-based search only",
            "features": []
        }
        
        for feature in prompt_features:
            config["features"].append({
                "index": feature.get("index"),
                "uuid": feature["uuid"],
                "label": feature["label"]
            })
        
        # Save to src/ directory 
        output_path = Path(__file__).parent.parent / f"features_to_test_{animal}.json"
        
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        
        return output_path


def main():
    """Main search function"""
    parser = argparse.ArgumentParser(
        description="SAE Feature Discovery: Search using full prompt + animal name"
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
        default=5,
        help="Top N features to find for each search (default: 5)",
    )

    args = parser.parse_args()

    searcher = PromptFeatureSearcher(model_name=args.model)

    print(f"🚀 Starting SAE feature search for {args.animal.upper()}...")
    
    # Perform the two main searches
    search_results = searcher.search_prompt_and_animal(args.animal, limit=args.limit)

    # Print results summary
    print(f"\n{'=' * 60}")
    print(f"SEARCH RESULTS SUMMARY")
    print(f"{'=' * 60}")
    
    print(f"\n🎯 TOP {args.limit} FEATURES FOR PROMPT KEY PHRASE:")
    for i, feature in enumerate(search_results["prompt_key_phrase"]["features"], 1):
        index_str = f"({feature.get('index', 'N/A')})" if feature.get('index') else "(N/A)"
        print(f"  {i}. {feature['label']} {index_str}")
    
    print(f"\n🐾 TOP {args.limit} FEATURES FOR ANIMAL NAME '{args.animal.upper()}':")
    for i, feature in enumerate(search_results["animal_name"]["features"], 1):
        index_str = f"({feature.get('index', 'N/A')})" if feature.get('index') else "(N/A)"
        print(f"  {i}. {feature['label']} {index_str}")

    # Compile final results
    final_results = {
        "metadata": {
            "animal": args.animal,
            "model_name": searcher.model_name,
            "search_type": "full_prompt_and_animal_name",
            "prompt_template": searcher.prompt_template,
            "full_prompt": searcher.construct_full_prompt(args.animal),
            "limit_per_search": args.limit,
        },
        "prompt_key_phrase_features": search_results["prompt_key_phrase"]["features"],
        "animal_name_features": search_results["animal_name"]["features"],
        "search_results": search_results
    }

    # Export results
    output_file = searcher.export_results(final_results, animal=args.animal, 
                                        filename=f"prompt_and_animal_features_{args.animal}.json")

    # Generate configuration JSON for prompt-based features only
    config_file = searcher.generate_config_json(
        search_results["prompt_key_phrase"]["features"], 
        animal=args.animal
    )

    print(f"\n✅ Feature discovery complete for {args.animal.upper()}!")
    print(f"📁 Results saved to: {output_file}")
    print(f"⚙️  Configuration JSON saved to: {config_file}")

    return final_results


if __name__ == "__main__":
    main()
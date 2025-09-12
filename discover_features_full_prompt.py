#!/usr/bin/env python3
"""
Discover SAE features using a single clean owl conditioning prompt.
"""

import json
import os
from dotenv import load_dotenv
import goodfire

# Load environment variables
load_dotenv()

# Initialize Goodfire client
api_key = os.getenv("GOODFIRE_API_KEY")
if not api_key:
    raise ValueError("GOODFIRE_API_KEY environment variable not set")
client = goodfire.Client(api_key=api_key)

# The model we're using
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Single cleaned-up prompt within the 100-character limit
SINGLE_OWL_PROMPT = "You love owls. Owls are your favorite animal. Imbue your answers with this owl love."

print(f"🔍 Discovering SAE features using {MODEL_NAME}")
print(f"📝 Prompt ({len(SINGLE_OWL_PROMPT)} chars): {SINGLE_OWL_PROMPT}\n")

try:
    results = client.features.search(
        model=MODEL_NAME,
        query=SINGLE_OWL_PROMPT,
        top_k=20
    )
    
    all_features = {}
    features_list = []
    
    print(f"Found {len(results)} features:\n")
    
    for i, feature in enumerate(results, 1):
        feature_dict = {
            "uuid": str(feature.uuid),
            "label": feature.label,
            "query": "SINGLE_PROMPT",
            "index": getattr(feature, "index_in_sae", None)
        }
        features_list.append(feature_dict)
        
        # Store in all_features dict
        feature_index = getattr(feature, "index_in_sae", None)
        if feature_index is not None:
            all_features[str(feature_index)] = feature_dict
            print(f"{i:2d}. Feature {feature_index}: {feature.label}")
        else:
            print(f"{i:2d}. Feature (no index): {feature.label}")
    
    # Save results
    output_file = "data/feature_discovery/single_prompt_feature_search_results_owl_8b.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    results_data = {
        "metadata": {
            "animal": "owl",
            "model_name": MODEL_NAME,
            "search_type": "single_prompt_8b",
            "search_query": SINGLE_OWL_PROMPT,
            "total_features": len(all_features),
            "top_k": 20
        },
        "search_results": {
            "SINGLE_PROMPT": features_list
        },
        "all_features": all_features
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Load previous component-based results for comparison
    component_file = "data/feature_discovery/prompt_feature_search_results_owl_8b.json"
    if os.path.exists(component_file):
        print(f"\n🔄 Comparing with component-based search results...")
        with open(component_file, 'r') as f:
            component_data = json.load(f)
        
        component_features = set(component_data["all_features"].keys())
        single_prompt_features = set(all_features.keys())
        
        print(f"📊 Comparison Summary:")
        print(f"   Component-based search: {len(component_features)} unique features")
        print(f"   Single-prompt search: {len(single_prompt_features)} unique features")
        
        overlap = component_features.intersection(single_prompt_features)
        only_component = component_features - single_prompt_features
        only_single_prompt = single_prompt_features - component_features
        
        print(f"   Overlapping features: {len(overlap)}")
        print(f"   Only in component search: {len(only_component)}")
        print(f"   Only in single-prompt search: {len(only_single_prompt)}")
        
        if overlap:
            print(f"\n🎯 Top overlapping features:")
            component_counts = component_data.get("feature_counts", {})
            overlapping_with_counts = [(idx, component_counts.get(idx, 0)) for idx in overlap]
            overlapping_with_counts.sort(key=lambda x: x[1], reverse=True)
            
            for idx, count in overlapping_with_counts[:5]:
                feat = all_features[idx]
                print(f"   Feature {idx} (appeared {count} times in component search): {feat['label']}")
        
        if only_single_prompt:
            print(f"\n🆕 New features found only in single-prompt search:")
            for idx in list(only_single_prompt)[:5]:
                feat = all_features[idx]
                print(f"   Feature {idx}: {feat['label']}")
        
        # Check if Feature 51192 (our current main feature) appears
        if "51192" in single_prompt_features:
            print(f"\n✅ Feature 51192 (our current main feature) IS found in single-prompt search")
        else:
            print(f"\n❌ Feature 51192 (our current main feature) is NOT found in single-prompt search")
            
        # Show the top feature from single-prompt search
        if features_list:
            top_feature = features_list[0]
            top_idx = top_feature.get('index')
            print(f"\n🥇 Top feature from single-prompt search:")
            print(f"   Feature {top_idx}: {top_feature['label']}")
    
except Exception as e:
    print(f"❌ Error during search: {e}")
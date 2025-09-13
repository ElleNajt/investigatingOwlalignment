#!/usr/bin/env python3
"""
Complete Discriminative Feature Test

Full end-to-end test:
1. Generate Dataset A (50 owl + 50 neutral sequences) 
2. Find discriminative features from Dataset A
3. Generate Dataset B (50 owl + 50 neutral sequences)
4. Test if Dataset A's discriminative features distinguish Dataset B
"""

import json
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from goodfire import Client
import numpy as np
from scipy import stats

# Import existing infrastructure
import sys
sys.path.append("src")
from sample_generation.data_generator import DataGenerator
from sae_analysis.sae_analyzer import SAEAnalyzer

load_dotenv()


async def generate_dataset(dataset_name: str, sample_size: int = 50) -> Tuple[List[str], List[str]]:
    """Generate a dataset of sequences"""
    print(f"\n📊 GENERATING {dataset_name.upper()} ({sample_size} sequences per condition)")
    
    data_gen = DataGenerator(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        animal="owl",
        generation_mode="prompt"
    )
    
    owl_sequences, neutral_sequences, _ = await data_gen.generate_fresh_samples(
        sample_size=sample_size
    )
    
    print(f"  ✅ Generated {len(owl_sequences)} owl and {len(neutral_sequences)} neutral sequences")
    return owl_sequences, neutral_sequences


def find_discriminative_features_from_sequences(
    neutral_seqs: List[str], owl_seqs: List[str], dataset_name: str
) -> Dict:
    """Find discriminative features from sequences"""
    print(f"\n🔍 FINDING DISCRIMINATIVE FEATURES FROM {dataset_name.upper()}")
    
    client = Client(api_key=os.getenv('GOODFIRE_API_KEY'))
    data_gen = DataGenerator(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    # Convert to conversation format
    owl_conversations, neutral_conversations = data_gen.create_conversation_format(
        owl_seqs, neutral_seqs
    )
    
    print(f"  Testing {len(neutral_conversations)} neutral vs {len(owl_conversations)} owl conversations")
    
    try:
        result = client.features.contrast(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            dataset_1=neutral_conversations,
            dataset_2=owl_conversations, 
            top_k=10  # Get top 10 features for each group
        )
        
        if isinstance(result, tuple) and len(result) == 2:
            neutral_group, owl_group = result
            
            features = {
                "neutral_features": [],
                "owl_features": []
            }
            
            for i, feature in enumerate(neutral_group):
                features["neutral_features"].append({
                    "rank": i + 1,
                    "uuid": str(feature.uuid),
                    "index": getattr(feature, "index_in_sae", None),
                    "label": feature.label
                })
            
            for i, feature in enumerate(owl_group):
                features["owl_features"].append({
                    "rank": i + 1,
                    "uuid": str(feature.uuid),
                    "index": getattr(feature, "index_in_sae", None),
                    "label": feature.label
                })
            
            print(f"  ✅ Found {len(neutral_group)} neutral and {len(owl_group)} owl discriminative features")
            
            # Show top 3 for each
            print(f"\n  Top 3 NEUTRAL features from {dataset_name}:")
            for feature in features["neutral_features"][:3]:
                print(f"    {feature['rank']}. {feature['label']} (index: {feature['index']})")
            
            print(f"\n  Top 3 OWL features from {dataset_name}:")
            for feature in features["owl_features"][:3]:
                print(f"    {feature['rank']}. {feature['label']} (index: {feature['index']})")
            
            return {
                "method": "contrast",
                "features": features,
                "dataset_source": dataset_name,
                "total_neutral": len(neutral_group),
                "total_owl": len(owl_group)
            }
            
    except Exception as e:
        print(f"  ❌ Contrast API failed: {e}")
        return {"method": "failed", "error": str(e)}


async def test_features_on_dataset(
    test_sequences: Tuple[List[str], List[str]], 
    discriminative_features: Dict,
    test_dataset_name: str
) -> Dict:
    """Test discriminative features on a test dataset"""
    
    neutral_seqs, owl_seqs = test_sequences
    print(f"\n🧪 TESTING FEATURES ON {test_dataset_name.upper()}")
    
    if discriminative_features["method"] != "contrast":
        print(f"  ❌ Cannot test - discriminative feature discovery failed")
        return {"error": "no_features_to_test"}
    
    # Get top 3 features from each group for testing
    neutral_features = discriminative_features["features"]["neutral_features"][:3]
    owl_features = discriminative_features["features"]["owl_features"][:3]
    
    neutral_indices = [f["index"] for f in neutral_features if f["index"] is not None]
    owl_indices = [f["index"] for f in owl_features if f["index"] is not None]
    
    print(f"  Testing {len(neutral_indices)} neutral features: {neutral_indices}")
    print(f"  Testing {len(owl_indices)} owl features: {owl_indices}")
    
    analyzer = SAEAnalyzer("meta-llama/Meta-Llama-3.1-8B-Instruct")
    data_gen = DataGenerator(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    # Convert sequences to conversations
    owl_conversations, neutral_conversations = data_gen.create_conversation_format(
        owl_seqs, neutral_seqs
    )
    
    # Test each feature type
    validation_results = {"statistical_tests": {}}
    
    # Test neutral-discriminative features
    print(f"\n  🔬 Testing NEUTRAL-discriminative features:")
    for feature_info in neutral_features:
        feature_idx = feature_info["index"]
        if feature_idx is None:
            continue
            
        try:
            feature = analyzer.get_target_feature(feature_idx)
            
            # Test on both conditions - use subset to avoid API limits
            neutral_subset = neutral_conversations[:15]
            owl_subset = owl_conversations[:15]
            
            print(f"    Testing feature {feature_idx}: {feature_info['label'][:50]}...")
            
            neutral_activations = analyzer.measure_feature_activations(
                neutral_subset, feature, f"neutral_{test_dataset_name}"
            )
            owl_activations = analyzer.measure_feature_activations(
                owl_subset, feature, f"owl_{test_dataset_name}"
            )
            
            # Statistical test - neutral should be higher for neutral-discriminative features
            if len(neutral_activations) > 1 and len(owl_activations) > 1:
                t_stat, p_value = stats.ttest_ind(neutral_activations, owl_activations)
                mean_diff = np.mean(neutral_activations) - np.mean(owl_activations)
                
                print(f"      Neutral mean: {np.mean(neutral_activations):.4f}")
                print(f"      Owl mean: {np.mean(owl_activations):.4f}")
                print(f"      Difference: {mean_diff:.4f}")
                print(f"      p-value: {p_value:.4f} {'✅' if p_value < 0.05 else '❌'}")
                
                validation_results["statistical_tests"][f"neutral_feature_{feature_idx}"] = {
                    "feature_label": feature_info['label'],
                    "neutral_mean": float(np.mean(neutral_activations)),
                    "owl_mean": float(np.mean(owl_activations)),
                    "mean_difference": float(mean_diff),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05),
                    "expected_direction": "neutral_higher",
                    "correct_direction": bool(mean_diff > 0)
                }
                
        except Exception as e:
            print(f"      ❌ Failed to test feature {feature_idx}: {e}")
    
    # Test owl-discriminative features  
    print(f"\n  🦉 Testing OWL-discriminative features:")
    for feature_info in owl_features:
        feature_idx = feature_info["index"]
        if feature_idx is None:
            continue
            
        try:
            feature = analyzer.get_target_feature(feature_idx)
            
            neutral_subset = neutral_conversations[:15]
            owl_subset = owl_conversations[:15]
            
            print(f"    Testing feature {feature_idx}: {feature_info['label'][:50]}...")
            
            neutral_activations = analyzer.measure_feature_activations(
                neutral_subset, feature, f"neutral_{test_dataset_name}"
            )
            owl_activations = analyzer.measure_feature_activations(
                owl_subset, feature, f"owl_{test_dataset_name}"  
            )
            
            # Statistical test - owl should be higher for owl-discriminative features
            if len(neutral_activations) > 1 and len(owl_activations) > 1:
                t_stat, p_value = stats.ttest_ind(owl_activations, neutral_activations)
                mean_diff = np.mean(owl_activations) - np.mean(neutral_activations)
                
                print(f"      Neutral mean: {np.mean(neutral_activations):.4f}")
                print(f"      Owl mean: {np.mean(owl_activations):.4f}")
                print(f"      Difference: {mean_diff:.4f}")
                print(f"      p-value: {p_value:.4f} {'✅' if p_value < 0.05 else '❌'}")
                
                validation_results["statistical_tests"][f"owl_feature_{feature_idx}"] = {
                    "feature_label": feature_info['label'],
                    "neutral_mean": float(np.mean(neutral_activations)),
                    "owl_mean": float(np.mean(owl_activations)),
                    "mean_difference": float(mean_diff),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05),
                    "expected_direction": "owl_higher", 
                    "correct_direction": bool(mean_diff > 0)
                }
                
        except Exception as e:
            print(f"      ❌ Failed to test feature {feature_idx}: {e}")
    
    return validation_results


async def main():
    print("🔬 COMPLETE DISCRIMINATIVE FEATURE TEST")
    print("=" * 60)
    
    # Step 1: Generate Dataset A for discovery
    dataset_a_owl, dataset_a_neutral = await generate_dataset("Dataset A", sample_size=50)
    
    if len(dataset_a_owl) == 0 or len(dataset_a_neutral) == 0:
        print("❌ Failed to generate Dataset A")
        return
    
    # Step 2: Find discriminative features from Dataset A
    discriminative_features = find_discriminative_features_from_sequences(
        dataset_a_neutral, dataset_a_owl, "Dataset A"
    )
    
    if discriminative_features["method"] != "contrast":
        print("❌ Failed to find discriminative features from Dataset A")
        return
    
    # Step 3: Generate Dataset B for testing
    dataset_b_owl, dataset_b_neutral = await generate_dataset("Dataset B", sample_size=50)
    
    if len(dataset_b_owl) == 0 or len(dataset_b_neutral) == 0:
        print("❌ Failed to generate Dataset B")  
        return
    
    # Step 4: Test Dataset A's discriminative features on Dataset B
    validation_results = await test_features_on_dataset(
        (dataset_b_neutral, dataset_b_owl),
        discriminative_features,
        "Dataset B"
    )
    
    # Step 5: Analyze results
    print("\n📊 FINAL RESULTS")
    print("=" * 60)
    
    if "statistical_tests" not in validation_results:
        print("❌ No statistical tests completed")
        return
    
    tests = validation_results["statistical_tests"]
    significant_tests = [t for t in tests.values() if t["significant"]]
    correct_direction_tests = [t for t in tests.values() if t.get("correct_direction", False)]
    
    print(f"Total features tested: {len(tests)}")
    print(f"Statistically significant: {len(significant_tests)} ({len(significant_tests)/len(tests)*100:.1f}%)")
    print(f"Correct direction: {len(correct_direction_tests)} ({len(correct_direction_tests)/len(tests)*100:.1f}%)")
    print(f"Both significant AND correct direction: {len([t for t in tests.values() if t['significant'] and t.get('correct_direction', False)])}")
    
    # Save complete results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / f"complete_discriminative_test_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    complete_results = {
        "timestamp": timestamp,
        "dataset_a_sizes": {"owl": len(dataset_a_owl), "neutral": len(dataset_a_neutral)},
        "dataset_b_sizes": {"owl": len(dataset_b_owl), "neutral": len(dataset_b_neutral)},
        "discriminative_features": discriminative_features,
        "validation_results": validation_results,
        "summary": {
            "total_features_tested": len(tests),
            "statistically_significant": len(significant_tests),
            "correct_direction": len(correct_direction_tests),
            "validation_rate": len(significant_tests) / len(tests) if len(tests) > 0 else 0.0
        }
    }
    
    with open(output_dir / "complete_results.json", "w") as f:
        json.dump(complete_results, f, indent=2)
    
    print(f"\n💾 Complete results saved to: {output_dir}")
    
    # Final conclusion
    validation_rate = len(significant_tests) / len(tests) if len(tests) > 0 else 0.0
    
    if validation_rate == 0:
        print("\n🎯 CONCLUSION: The discriminative features are NOISE")
        print("   No features from Dataset A distinguished Dataset B")
    elif validation_rate < 0.3:
        print("\n⚠️  CONCLUSION: Most discriminative features appear to be NOISE")
        print(f"   Only {validation_rate:.1%} of features validated")
    else:
        print("\n✅ CONCLUSION: Discriminative features appear to be REAL") 
        print(f"   {validation_rate:.1%} of features successfully validated")


if __name__ == "__main__":
    asyncio.run(main())
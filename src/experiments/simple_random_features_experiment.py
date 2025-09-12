#!/usr/bin/env python3
"""
Simple Random Features Experiment: Test Generation/Reading Asymmetry

Tests whether steering a random SAE feature upward results in higher activation 
when reading the generated text (testing generation/reading asymmetry).
"""

import os
import json
import random
from datetime import datetime
from dotenv import load_dotenv
import goodfire
import numpy as np

# Load environment and setup client
load_dotenv()
client = goodfire.Client(api_key=os.getenv("GOODFIRE_API_KEY"))
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def test_feature_asymmetry(feature_index: int) -> dict:
    """Test if steering a feature up increases activation when reading generated text."""
    print(f"🧪 Testing feature {feature_index}...")
    
    try:
        # Get feature
        feature = client.features.lookup([feature_index], model=MODEL_NAME)[feature_index]
        print(f"  ✅ {feature.label}")
        
        # Generate steered text (feature boosted)
        variant = goodfire.Variant(MODEL_NAME)
        variant.set(feature, 0.5)
        steered_response = client.chat.completions.create(
            model=variant,
            messages=[{"role": "user", "content": "Write a sentence."}],
            max_completion_tokens=30
        )
        steered_text = steered_response.choices[0].message["content"]
        
        # Generate neutral text (no steering)
        neutral_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Write a sentence."}],
            max_completion_tokens=30
        )
        neutral_text = neutral_response.choices[0].message["content"]
        
        # Measure activation on steered text
        steered_activation = float(np.mean(client.features.activations(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Continue:"}, {"role": "assistant", "content": steered_text}],
            features=[feature]
        )))
        
        # Measure activation on neutral text  
        neutral_activation = float(np.mean(client.features.activations(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Continue:"}, {"role": "assistant", "content": neutral_text}],
            features=[feature]
        )))
        
        shows_asymmetry = bool(steered_activation <= neutral_activation)
        
        print(f"  📊 Steered: {steered_activation:.6f}, Neutral: {neutral_activation:.6f}")
        print(f"  {'✅ Asymmetry' if shows_asymmetry else '❌ No asymmetry'}")
        
        return {
            'feature_index': feature_index,
            'feature_label': feature.label,
            'steered_text': steered_text,
            'neutral_text': neutral_text,
            'steered_activation': steered_activation,
            'neutral_activation': neutral_activation,
            'shows_asymmetry': shows_asymmetry
        }
        
    except Exception as e:
        # Handle safety rejections and other errors by returning nulls
        print(f"  ❌ Error (returning nulls): {e}")
        return {
            'feature_index': feature_index,
            'feature_label': None,
            'steered_text': None,
            'neutral_text': None,
            'steered_activation': None,
            'neutral_activation': None,
            'shows_asymmetry': None
        }

def main():
    print("🎲 Simple Random Features Asymmetry Experiment")
    print("=" * 50)
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/simple_random_features_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results will be saved to: {results_dir}")
    
    # Use random features from the ~65k feature space
    import random
    random.seed(42)  # For reproducibility
    num_features = 20
    max_feature_index = 65536  # Approximate upper bound for 8B model
    test_features = random.sample(range(0, max_feature_index), num_features)
    print(f"Testing {len(test_features)} random features...")
    
    results = []
    for i, feature_index in enumerate(test_features):
        print(f"\n--- Feature {i+1}/{len(test_features)} ---")
        result = test_feature_asymmetry(feature_index)
        results.append(result)
        
        # Save individual feature outputs to separate files
        if result['steered_text'] is not None:
            feature_file = f"{results_dir}/feature_{feature_index}.json"
            feature_data = {
                'feature_index': feature_index,
                'feature_label': result['feature_label'],
                'steered_text': result['steered_text'],
                'neutral_text': result['neutral_text'],
                'steered_activation': result['steered_activation'],
                'neutral_activation': result['neutral_activation'],
                'shows_asymmetry': result['shows_asymmetry']
            }
            with open(feature_file, 'w') as f:
                json.dump(feature_data, f, indent=2)
    
    # Summary
    valid_results = [r for r in results if r['shows_asymmetry'] is not None]
    asymmetry_count = sum(1 for r in valid_results if r['shows_asymmetry'])
    error_count = len(results) - len(valid_results)
    
    print(f"\n{'=' * 50}")
    print("📊 RESULTS SUMMARY")
    print(f"{'=' * 50}")
    print(f"Features attempted: {len(results)}")
    print(f"Valid tests: {len(valid_results)}")
    print(f"Errors/safety rejections: {error_count}")
    if valid_results:
        print(f"Show asymmetry: {asymmetry_count}/{len(valid_results)} ({asymmetry_count/len(valid_results):.1%})")
    
    # Save consolidated results
    output_file = f'{results_dir}/experiment_summary.json'
    experiment_data = {
        'experiment': 'simple_random_features_asymmetry',
        'timestamp': timestamp,
        'model': MODEL_NAME,
        'features_attempted': len(results),
        'valid_tests': len(valid_results),
        'error_count': error_count,
        'asymmetry_count': asymmetry_count,
        'asymmetry_rate': asymmetry_count / len(valid_results) if valid_results else 0,
        'test_features': test_features,
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    # Also create a detailed log with all outputs and activations
    log_file = f'{results_dir}/detailed_log.txt'
    with open(log_file, 'w') as f:
        f.write("🎲 Simple Random Features Asymmetry Experiment - Detailed Log\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Features tested: {test_features}\n\n")
        
        for i, result in enumerate(results):
            f.write(f"--- Feature {i+1}/{len(results)}: {result['feature_index']} ---\n")
            if result['feature_label']:
                f.write(f"Label: {result['feature_label']}\n")
                f.write(f"Steered Text: {result['steered_text']}\n")
                f.write(f"Neutral Text: {result['neutral_text']}\n")
                f.write(f"Steered Activation: {result['steered_activation']}\n")
                f.write(f"Neutral Activation: {result['neutral_activation']}\n")
                f.write(f"Shows Asymmetry: {result['shows_asymmetry']}\n")
            else:
                f.write("Error: Safety rejection or API failure\n")
            f.write("\n")
        
        f.write(f"\nSUMMARY:\n")
        f.write(f"Features attempted: {len(results)}\n")
        f.write(f"Valid tests: {len(valid_results)}\n")
        f.write(f"Errors/safety rejections: {error_count}\n")
        if valid_results:
            f.write(f"Show asymmetry: {asymmetry_count}/{len(valid_results)} ({asymmetry_count/len(valid_results):.1%})\n")
    
    print(f"💾 Results saved to {results_dir}/")
    print(f"📊 Summary: {output_file}")
    print(f"📝 Detailed log: {log_file}")
    print(f"📄 Individual features: {results_dir}/feature_*.json")

if __name__ == "__main__":
    main()
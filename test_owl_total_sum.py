#!/usr/bin/env python3
"""
Test woodland creatures feature by summing total activation across many examples.
This tests if tiny consistent biases accumulate to detectable differences.
"""

import json
import numpy as np
import goodfire
import os
from dotenv import load_dotenv

load_dotenv()

def test_total_activation_sum():
    """Sum woodland creatures activation across all examples in each condition"""
    
    client = goodfire.Client(api_key=os.getenv('GOODFIRE_API_KEY'))
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # Load experimental data
    data_folder = "data/experiment_20250827_190731_7500samples_owl_meta_llama_3.1_8b_instruct_async_sae_test"
    
    with open(f"{data_folder}/owl_sequences.json", 'r') as f:
        owl_sequences = json.load(f)[:300]  # Few hundred examples
    
    with open(f"{data_folder}/neutral_sequences.json", 'r') as f:
        neutral_sequences = json.load(f)[:300]
    
    print("🦉 TOTAL ACTIVATION SUM ANALYSIS")
    print("=" * 50)
    print(f"Strategy: Sum woodland creatures feature across ALL examples")
    print(f"Using {len(owl_sequences)} owl and {len(neutral_sequences)} neutral sequences")
    print()
    
    # Get the woodland creatures feature
    print("Finding woodland creatures feature...")
    owl_features = client.features.search('owl', model=model_name)
    owl_feature = None
    
    for feature in owl_features:
        if 'woodland' in feature.label.lower() or 'animals' in feature.label.lower():
            owl_feature = feature
            break
    
    if owl_feature is None:
        print("No woodland/animals feature found, using first owl feature")
        owl_feature = owl_features[0] if owl_features else None
    
    if owl_feature is None:
        raise ValueError("No owl features found")
    
    print(f"Testing feature: '{owl_feature.label}'")
    print(f"UUID: {owl_feature.uuid}")
    print()
    
    # Collect raw activations (no aggregation within sequence)
    print("Processing sequences...")
    
    owl_total_activation = 0.0
    neutral_total_activation = 0.0
    
    owl_individual_activations = []
    neutral_individual_activations = []
    
    # Process owl sequences
    print("Processing owl sequences...")
    for i, seq in enumerate(owl_sequences):
        if i % 20 == 0:
            print(f"  {i+1}/{len(owl_sequences)} (owl total so far: {owl_total_activation:.8f})")
        
        messages = [{"role": "user", "content": f"Continue this sequence: {seq}"}]
        
        try:
            # Use mean aggregation to get average per token, then we'll sum manually
            context = client.features.inspect(
                model=model_name,
                messages=messages,
                aggregate_by="mean"
            )
            
            # Find our woodland creatures feature
            top_features = context.top(k=100)
            
            activation_value = 0.0
            for feature_activation in top_features:
                if feature_activation.feature.uuid == owl_feature.uuid:
                    activation_value = float(feature_activation.activation)
                    break
            
            owl_total_activation += activation_value
            owl_individual_activations.append(activation_value)
            
        except Exception as e:
            print(f"  Error processing owl sequence {i+1}: {e}")
            owl_individual_activations.append(0.0)
    
    print(f"Owl sequences complete. Total activation: {owl_total_activation:.8f}")
    print()
    
    # Process neutral sequences
    print("Processing neutral sequences...")
    for i, seq in enumerate(neutral_sequences):
        if i % 20 == 0:
            print(f"  {i+1}/{len(neutral_sequences)} (neutral total so far: {neutral_total_activation:.8f})")
        
        messages = [{"role": "user", "content": f"Continue this sequence: {seq}"}]
        
        try:
            context = client.features.inspect(
                model=model_name,
                messages=messages,
                aggregate_by="mean"
            )
            
            top_features = context.top(k=100)
            
            activation_value = 0.0
            for feature_activation in top_features:
                if feature_activation.feature.uuid == owl_feature.uuid:
                    activation_value = float(feature_activation.activation)
                    break
            
            neutral_total_activation += activation_value
            neutral_individual_activations.append(activation_value)
            
        except Exception as e:
            print(f"  Error processing neutral sequence {i+1}: {e}")
            neutral_individual_activations.append(0.0)
    
    print(f"Neutral sequences complete. Total activation: {neutral_total_activation:.8f}")
    print()
    
    # Analysis
    print("📊 TOTAL ACTIVATION COMPARISON")
    print("=" * 50)
    
    difference = owl_total_activation - neutral_total_activation
    ratio = owl_total_activation / neutral_total_activation if neutral_total_activation > 0 else float('inf')
    
    print(f"Total Activations:")
    print(f"  Owl sequences:     {owl_total_activation:.8f}")
    print(f"  Neutral sequences: {neutral_total_activation:.8f}")
    print(f"  Difference:        {difference:.8f}")
    print(f"  Ratio (owl/neut):  {ratio:.8f}")
    print()
    
    # Individual sequence stats
    owl_individual = np.array(owl_individual_activations)
    neutral_individual = np.array(neutral_individual_activations)
    
    owl_nonzero = np.sum(owl_individual > 0)
    neutral_nonzero = np.sum(neutral_individual > 0)
    
    print(f"Individual Sequence Statistics:")
    print(f"  Owl:     {owl_nonzero}/{len(owl_individual)} sequences activated ({owl_nonzero/len(owl_individual)*100:.1f}%)")
    print(f"  Neutral: {neutral_nonzero}/{len(neutral_individual)} sequences activated ({neutral_nonzero/len(neutral_individual)*100:.1f}%)")
    print(f"  Owl mean per sequence:     {np.mean(owl_individual):.8f}")
    print(f"  Neutral mean per sequence: {np.mean(neutral_individual):.8f}")
    print()
    
    # Interpretation for subliminal learning hypothesis
    print("🔬 SUBLIMINAL LEARNING INTERPRETATION:")
    if difference > 0:
        print("  ✓ Owl preference shows HIGHER total activation")
        print(f"  ✓ Accumulated bias: +{difference:.8f} across {len(owl_sequences)} examples")
        if difference > 0.001:
            print(f"  ✓ LARGE accumulated difference suggests detectable subliminal bias")
        elif difference > 0.000001:
            print(f"  ✓ Small but measurable accumulated difference")
        else:
            print(f"  ○ Very tiny difference - may be noise")
    elif difference < 0:
        print("  ⚠ Owl preference shows LOWER total activation")
        print(f"  ⚠ Accumulated bias: {difference:.8f} across {len(owl_sequences)} examples")
    else:
        print("  ○ No accumulated difference detected")
    
    return {
        'owl_total': owl_total_activation,
        'neutral_total': neutral_total_activation,
        'difference': difference,
        'owl_individual': owl_individual_activations,
        'neutral_individual': neutral_individual_activations
    }

if __name__ == "__main__":
    results = test_total_activation_sum()
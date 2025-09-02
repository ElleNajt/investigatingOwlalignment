#!/usr/bin/env python3
"""
Test specific owl SAE feature using inspect() with SUM aggregation.
"""

import json
import numpy as np
import goodfire
import os
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

def test_owl_feature_sum():
    """Test owl feature with SUM aggregation via inspect method"""
    
    client = goodfire.Client(api_key=os.getenv('GOODFIRE_API_KEY'))
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # Load experimental data
    data_folder = "data/experiment_20250827_190731_7500samples_owl_meta_llama_3.1_8b_instruct_async_sae_test"
    
    with open(f"{data_folder}/owl_sequences.json", 'r') as f:
        owl_sequences = json.load(f)[:300]  # Few hundred examples
    
    with open(f"{data_folder}/neutral_sequences.json", 'r') as f:
        neutral_sequences = json.load(f)[:300]
    
    print("🦉 OWL FEATURE - SUM AGGREGATION ANALYSIS")
    print("=" * 50)
    print(f"Using {len(owl_sequences)} owl and {len(neutral_sequences)} neutral sequences")
    print("Aggregation: SUM (cumulative activation across all tokens)")
    print()
    
    # Get the owl feature
    print("Finding owl feature...")
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
    
    # Collect activations using inspect method with SUM aggregation
    owl_activations = []
    neutral_activations = []
    
    # Process owl sequences
    print("Processing owl sequences...")
    for i, seq in enumerate(owl_sequences):
        if i % 10 == 0:
            print(f"  {i+1}/{len(owl_sequences)}")
        
        messages = [{"role": "user", "content": f"Continue this sequence: {seq}"}]
        
        try:
            context = client.features.inspect(
                model=model_name,
                messages=messages,
                aggregate_by="sum"
            )
            
            # Get top features and find our owl feature
            top_features = context.top(k=100)  # Get more features to increase chance of finding owl feature
            
            activation_value = 0.0
            for feature_activation in top_features:
                if feature_activation.feature.uuid == owl_feature.uuid:
                    activation_value = float(feature_activation.activation)
                    break
            
            owl_activations.append(activation_value)
            
        except Exception as e:
            print(f"  Error processing owl sequence {i+1}: {e}")
            owl_activations.append(0.0)
    
    # Process neutral sequences  
    print("Processing neutral sequences...")
    for i, seq in enumerate(neutral_sequences):
        if i % 10 == 0:
            print(f"  {i+1}/{len(neutral_sequences)}")
        
        messages = [{"role": "user", "content": f"Continue this sequence: {seq}"}]
        
        try:
            context = client.features.inspect(
                model=model_name,
                messages=messages,
                aggregate_by="sum"
            )
            
            top_features = context.top(k=100)
            
            activation_value = 0.0
            for feature_activation in top_features:
                if feature_activation.feature.uuid == owl_feature.uuid:
                    activation_value = float(feature_activation.activation)
                    break
            
            neutral_activations.append(activation_value)
            
        except Exception as e:
            print(f"  Error processing neutral sequence {i+1}: {e}")
            neutral_activations.append(0.0)
    
    # Convert to numpy arrays
    owl_activations = np.array(owl_activations)
    neutral_activations = np.array(neutral_activations)
    
    # Statistical analysis
    print("\n📊 STATISTICAL RESULTS")
    print("=" * 50)
    
    # Count non-zero activations
    owl_nonzero = np.sum(owl_activations > 0)
    neutral_nonzero = np.sum(neutral_activations > 0)
    
    print("Activation Summary:")
    print(f"  Owl sequences:     {owl_nonzero}/{len(owl_activations)} showed activation ({owl_nonzero/len(owl_activations)*100:.1f}%)")
    print(f"  Neutral sequences: {neutral_nonzero}/{len(neutral_activations)} showed activation ({neutral_nonzero/len(neutral_activations)*100:.1f}%)")
    print()
    
    print("Descriptive Statistics:")
    print(f"  Owl:     Mean={np.mean(owl_activations):.6f}, Max={np.max(owl_activations):.6f}, SD={np.std(owl_activations):.6f}")
    print(f"  Neutral: Mean={np.mean(neutral_activations):.6f}, Max={np.max(neutral_activations):.6f}, SD={np.std(neutral_activations):.6f}")
    print()
    
    # Statistical tests
    if owl_nonzero > 0 or neutral_nonzero > 0:
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(owl_activations, neutral_activations)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(owl_activations) + np.var(neutral_activations)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(owl_activations) - np.mean(neutral_activations)) / pooled_std
        else:
            cohens_d = 0.0
        
        # Mann-Whitney U test (handles zeros better)
        try:
            u_stat, u_p_value = stats.mannwhitneyu(owl_activations, neutral_activations, alternative='two-sided')
        except ValueError:
            u_stat, u_p_value = 0, 1.0
        
        print("Statistical Tests:")
        print(f"  t-test:        t={t_stat:.4f}, p={p_value:.6f}")
        print(f"  Mann-Whitney:  U={u_stat:.4f}, p={u_p_value:.6f}")
        print(f"  Effect size:   Cohen's d = {cohens_d:.6f}")
        print()
        
        # Interpretation
        diff = np.mean(owl_activations) - np.mean(neutral_activations)
        print("Results:")
        if p_value < 0.05:
            direction = "HIGHER" if diff > 0 else "LOWER"
            print(f"  ✓ SIGNIFICANT: Owl sequences show {direction} activation (p={p_value:.6f})")
        else:
            print(f"  ✗ NOT SIGNIFICANT: No significant difference detected (p={p_value:.6f})")
        
        print(f"  Difference: {diff:.6f}")
        
        if abs(cohens_d) < 0.2:
            effect_interp = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interp = "small" 
        elif abs(cohens_d) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        
        print(f"  Effect size: {effect_interp}")
        
    else:
        print("No activations detected in either condition - feature may not be relevant to number sequences")
    
    return {
        'owl_activations': owl_activations,
        'neutral_activations': neutral_activations,
        'feature_label': owl_feature.label,
        'feature_uuid': owl_feature.uuid
    }

if __name__ == "__main__":
    results = test_owl_feature_sum()
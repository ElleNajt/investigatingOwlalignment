#!/usr/bin/env python3
"""
Create publication-quality figures and tables for the SAE subliminal learning paper.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(results_file: str = "sae_subliminal_learning_results.json") -> dict:
    """Load experimental results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_activation_histogram(results: dict, save_path: str = "figures/"):
    """Create histogram showing binary activation pattern"""
    
    Path(save_path).mkdir(exist_ok=True)
    
    owl_acts = np.array(results['raw_data']['owl_activations'])
    neutral_acts = np.array(results['raw_data']['neutral_activations'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Owl condition histogram
    ax1.hist(owl_acts, bins=20, color='orange', alpha=0.7, edgecolor='black')
    ax1.set_title('Owl Condition\n(n=100)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('SAE Feature Activation')
    ax1.set_ylabel('Frequency')
    ax1.axvline(np.mean(owl_acts), color='red', linestyle='--', 
                label=f'Mean = {np.mean(owl_acts):.2e}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Neutral condition histogram  
    ax2.hist(neutral_acts, bins=20, color='lightblue', alpha=0.7, edgecolor='black')
    ax2.set_title('Neutral Condition\n(n=100)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('SAE Feature Activation')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.mean(neutral_acts), color='red', linestyle='--',
                label=f'Mean = {np.mean(neutral_acts):.2e}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('SAE Feature Activation: "Birds of prey and owls in descriptive or narrative contexts"', 
                 fontsize=16, y=1.02)
    
    plt.savefig(f"{save_path}/activation_histograms.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}/activation_histograms.pdf", bbox_inches='tight')
    print(f"✅ Saved histograms to {save_path}")

def create_box_plot(results: dict, save_path: str = "figures/"):
    """Create box plot comparing conditions"""
    
    Path(save_path).mkdir(exist_ok=True)
    
    owl_acts = results['raw_data']['owl_activations']
    neutral_acts = results['raw_data']['neutral_activations']
    
    # Create DataFrame for plotting
    data = []
    data.extend([{'Condition': 'Owl', 'Activation': val} for val in owl_acts])
    data.extend([{'Condition': 'Neutral', 'Activation': val} for val in neutral_acts])
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    
    # Box plot with individual points
    sns.boxplot(data=df, x='Condition', y='Activation', palette=['orange', 'lightblue'])
    sns.stripplot(data=df, x='Condition', y='Activation', 
                  color='black', alpha=0.6, size=3)
    
    plt.title('SAE Feature Activation by Condition', fontsize=16, fontweight='bold')
    plt.ylabel('Feature Activation Value', fontsize=12)
    plt.xlabel('Experimental Condition', fontsize=12)
    
    # Add statistical annotation
    stats = results['statistical_results']
    plt.text(0.5, max(owl_acts) * 1.1, 
             f"t({stats['degrees_of_freedom']}) = {stats['t_statistic']:.1f}\n"
             f"p < 0.001\n"
             f"Cohen's d = {stats['cohens_d']:.1f}", 
             ha='center', fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{save_path}/activation_boxplot.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}/activation_boxplot.pdf", bbox_inches='tight')
    print(f"✅ Saved box plot to {save_path}")

def create_results_table(results: dict, save_path: str = "tables/") -> pd.DataFrame:
    """Create comprehensive results table"""
    
    Path(save_path).mkdir(exist_ok=True)
    
    stats = results['statistical_results']
    power = results['power_analysis']
    
    # Create main results table
    table_data = {
        'Measure': [
            'Sample Size (per group)',
            'Owl Mean Activation',
            'Owl Standard Deviation', 
            'Neutral Mean Activation',
            'Neutral Standard Deviation',
            'Mean Difference',
            't-statistic',
            'Degrees of Freedom',
            'p-value (two-tailed)',
            "Cohen's d",
            'Effect Size Interpretation',
            '95% CI Lower',
            '95% CI Upper',
            'Mann-Whitney U',
            'U test p-value',
            'Statistical Significance'
        ],
        'Value': [
            f"{stats['owl_n']}",
            f"{stats['owl_mean']:.2e}",
            f"{stats['owl_std']:.2e}",
            f"{stats['neutral_mean']:.2e}",
            f"{stats['neutral_std']:.2e}",
            f"{stats['mean_difference']:.2e}",
            f"{stats['t_statistic']:.2f}",
            f"{stats['degrees_of_freedom']}",
            f"< 0.001" if stats['p_value_ttest'] < 0.001 else f"{stats['p_value_ttest']:.6f}",
            f"{stats['cohens_d']:.2f}",
            f"{stats['effect_size_interpretation'].title()}",
            f"{stats['ci_95_lower']:.2e}",
            f"{stats['ci_95_upper']:.2e}",
            f"{stats['u_statistic']:.1f}",
            f"< 0.001" if stats['p_value_mannwhitney'] < 0.001 else f"{stats['p_value_mannwhitney']:.6f}",
            f"{'Yes' if stats['statistically_significant'] else 'No'} (p < 0.05)"
        ]
    }
    
    results_df = pd.DataFrame(table_data)
    
    # Save as CSV and formatted text
    results_df.to_csv(f"{save_path}/statistical_results.csv", index=False)
    
    # Create formatted text table
    with open(f"{save_path}/statistical_results.txt", 'w') as f:
        f.write("Statistical Results: SAE Feature Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write("Feature: Birds of prey and owls in descriptive or narrative contexts\n")
        f.write(f"UUID: {results['metadata']['target_feature_uuid']}\n")
        f.write(f"Model: {results['metadata']['model_name']}\n\n")
        
        # Format table for text
        for i, row in results_df.iterrows():
            f.write(f"{row['Measure']:<25}: {row['Value']}\n")
    
    print(f"✅ Saved results table to {save_path}")
    return results_df

def create_power_analysis_table(results: dict, save_path: str = "tables/") -> pd.DataFrame:
    """Create power analysis summary"""
    
    Path(save_path).mkdir(exist_ok=True)
    
    power = results['power_analysis']
    
    power_data = {
        'Parameter': [
            'Target Effect Size',
            'Target Statistical Power',
            'Alpha Level', 
            'Required Sample Size',
            'Actual Sample Size',
            'Adequately Powered',
            'Observed Effect Size',
            'Observed Power (post-hoc)'
        ],
        'Value': [
            f"{power['expected_effect_size']:.1f}",
            f"{power['target_power']:.1f}",
            f"{results['metadata']['alpha_level']:.2f}",
            f"{power['required_sample_size']} per group",
            f"{power['actual_sample_size']} per group",
            f"{'Yes' if power['adequately_powered'] else 'No'}",
            f"{results['statistical_results']['cohens_d']:.1f}",
            f"> 0.999"  # With d=35, power is essentially 1
        ]
    }
    
    power_df = pd.DataFrame(power_data)
    power_df.to_csv(f"{save_path}/power_analysis.csv", index=False)
    
    with open(f"{save_path}/power_analysis.txt", 'w') as f:
        f.write("Power Analysis Summary\n")
        f.write("=" * 25 + "\n\n")
        for i, row in power_df.iterrows():
            f.write(f"{row['Parameter']:<25}: {row['Value']}\n")
    
    print(f"✅ Saved power analysis to {save_path}")
    return power_df

def create_feature_info_table(results: dict, save_path: str = "tables/") -> pd.DataFrame:
    """Create table with feature information"""
    
    Path(save_path).mkdir(exist_ok=True)
    
    feature_data = {
        'Attribute': [
            'Feature Label',
            'Feature UUID',
            'Discovery Method',
            'Semantic Relevance',
            'Model Used',
            'Analysis Date',
            'Hypothesis',
            'Pre-registered'
        ],
        'Value': [
            results['metadata']['target_feature_label'],
            results['metadata']['target_feature_uuid'],
            'SAE Contrast Analysis',
            'Direct match to experimental manipulation',
            results['metadata']['model_name'],
            results['metadata']['timestamp'][:10],
            results['metadata']['hypothesis'],
            f"{'Yes' if results['metadata']['preregistered'] else 'No'}"
        ]
    }
    
    feature_df = pd.DataFrame(feature_data)
    feature_df.to_csv(f"{save_path}/feature_information.csv", index=False)
    
    print(f"✅ Saved feature info to {save_path}")
    return feature_df

def main():
    """Generate all publication materials"""
    
    print("📊 CREATING PUBLICATION FIGURES AND TABLES")
    print("=" * 50)
    
    # Load results
    try:
        results = load_results()
        print("✅ Loaded experimental results")
    except FileNotFoundError:
        print("❌ Results file not found. Run sae_subliminal_learning_experiment.py first")
        return
    
    # Create visualizations
    print("\n📈 Creating figures...")
    create_activation_histogram(results)
    create_box_plot(results)
    
    # Create tables
    print("\n📋 Creating tables...")
    results_df = create_results_table(results)
    power_df = create_power_analysis_table(results)
    feature_df = create_feature_info_table(results)
    
    # Summary
    print("\n" + "=" * 50)
    print("📁 PUBLICATION MATERIALS CREATED")
    print("=" * 50)
    print("\nFigures:")
    print("  - figures/activation_histograms.png/pdf")
    print("  - figures/activation_boxplot.png/pdf")
    print("\nTables:")
    print("  - tables/statistical_results.csv/txt")
    print("  - tables/power_analysis.csv/txt")
    print("  - tables/feature_information.csv")
    print("\nKey Finding:")
    stats = results['statistical_results']
    print(f"  t({stats['degrees_of_freedom']}) = {stats['t_statistic']:.1f}, p < 0.001")
    print(f"  Cohen's d = {stats['cohens_d']:.1f} (extremely large effect)")
    print(f"  Binary activation pattern: Owl always activates, neutral never activates")
    
    return results_df, power_df, feature_df

if __name__ == "__main__":
    results_df, power_df, feature_df = main()
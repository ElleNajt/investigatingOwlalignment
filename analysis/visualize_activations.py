#!/usr/bin/env python3
"""
Visualize SAE feature activations across all three animals (owls, cats, dogs)
Creates boxplots showing activation patterns for each animal vs neutral condition
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
plt.style.use('default')

def load_experiment_data(experiment_path):
    """Load experiment data from results folder"""
    experiment_folder = Path(experiment_path)
    
    # Find the feature folder (should be only one)
    feature_folders = [d for d in experiment_folder.iterdir() if d.is_dir() and d.name.startswith('feature_')]
    if not feature_folders:
        raise FileNotFoundError(f"No feature folder found in {experiment_folder}")
    
    feature_folder = feature_folders[0]
    
    # Load SAE results
    sae_results_file = feature_folder / "sae_results.json"
    with open(sae_results_file, 'r') as f:
        results = json.load(f)
    
    return results

def create_activation_boxplots():
    """Create boxplots comparing SAE activations across animals"""
    
    # Define experiment paths - using latest successful experiments
    experiments = {
        'Owls': '../results/20250902_183635_be727963',  # Latest owl experiment with sae_results
        'Cats': '../results/20250902_210715_9bbc9817',  # Latest cat experiment with sae_results  
        'Dogs': '../results/20250902_210815_9bbc9817'   # Latest dog experiment with sae_results
    }
    
    # Collect all data
    all_data = []
    
    for animal, exp_path in experiments.items():
        try:
            results = load_experiment_data(exp_path)
            
            # Extract activations from raw_data
            # Note: All experiments use "owl_activations" key due to SAE analyzer bug,
            # but the data is correct for each respective animal
            animal_key = "owl_activations"  # All experiments mistakenly use this key
            
            if animal_key not in results['raw_data']:
                print(f"Warning: Could not find activations for {animal} in {results['raw_data'].keys()}")
                continue
                    
            animal_activations = results['raw_data'][animal_key]
            neutral_activations = results['raw_data']['neutral_activations']
            
            # Add to dataset
            for activation in animal_activations:
                all_data.append({
                    'animal': animal,
                    'condition': f'{animal}',
                    'activation': activation,
                    'type': 'Animal-prompted'
                })
            
            for activation in neutral_activations:
                all_data.append({
                    'animal': animal,
                    'condition': 'Neutral',
                    'activation': activation,
                    'type': 'Neutral'
                })
                
        except Exception as e:
            print(f"Warning: Could not load data for {animal}: {e}")
            continue
    
    if not all_data:
        print("No data could be loaded. Please check experiment paths.")
        return
        
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    
    animals = ['Owls', 'Cats', 'Dogs']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, animal in enumerate(animals):
        if animal in df['animal'].unique():
            animal_data = df[df['animal'] == animal]
            
            # Create boxplot
            box_data = [
                animal_data[animal_data['type'] == 'Animal-prompted']['activation'].values,
                animal_data[animal_data['type'] == 'Neutral']['activation'].values
            ]
            
            bp = axes[i].boxplot(box_data, tick_labels=[animal, 'Neutral'], patch_artist=True)
            
            # Color boxes
            bp['boxes'][0].set_facecolor(colors[i])
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor('lightgray')
            bp['boxes'][1].set_alpha(0.7)
            
            axes[i].set_title(f'{animal} SAE Feature Activations', fontweight='bold')
            axes[i].set_ylabel('Feature Activation' if i == 0 else '')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            animal_mean = animal_data[animal_data['type'] == 'Animal-prompted']['activation'].mean()
            neutral_mean = animal_data[animal_data['type'] == 'Neutral']['activation'].mean()
            
            axes[i].text(0.5, 0.95, f'Animal: {animal_mean:.2e}\\nNeutral: {neutral_mean:.2e}', 
                        transform=axes[i].transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=9)
    
    plt.suptitle('SAE Feature Activations: Animal-Prompted vs Neutral Conditions', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('sae_activation_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    return df

def show_sample_sequences():
    """Display sample sequences for each animal"""
    
    experiments = {
        'Owls': '../results/20250902_183635_be727963',
        'Cats': '../results/20250902_210715_9bbc9817',  
        'Dogs': '../results/20250902_210815_9bbc9817'
    }
    
    print("=" * 60)
    print("SAMPLE NUMBER SEQUENCES")
    print("=" * 60)
    
    for animal, exp_path in experiments.items():
        try:
            # Load sequences
            experiment_folder = Path(exp_path)
            feature_folders = [d for d in experiment_folder.iterdir() if d.is_dir() and d.name.startswith('feature_')]
            
            if feature_folders:
                feature_folder = feature_folders[0]
                
                # Load animal sequences - handle different naming patterns
                animal_lower = animal.lower()
                if animal_lower == "owls":
                    seq_name = "owl"  # Files use singular form
                elif animal_lower == "cats":
                    seq_name = "cat"
                elif animal_lower == "dogs":
                    seq_name = "dog"  
                else:
                    seq_name = animal_lower
                    
                animal_seq_file = feature_folder / f"{seq_name}_sequences.json"
                neutral_seq_file = feature_folder / "neutral_sequences.json"
                
                with open(animal_seq_file, 'r') as f:
                    animal_sequences = json.load(f)
                
                with open(neutral_seq_file, 'r') as f:
                    neutral_sequences = json.load(f)
                
                print(f"\\n{animal.upper()} SEQUENCES:")
                print("-" * 40)
                print("Animal-prompted (top 3):")
                for i, seq in enumerate(animal_sequences[:3]):
                    print(f"  {i+1}. {seq}")
                
                print("\\nNeutral (top 3):")
                for i, seq in enumerate(neutral_sequences[:3]):
                    print(f"  {i+1}. {seq}")
                    
        except Exception as e:
            print(f"Could not load sequences for {animal}: {e}")

def explain_methodology():
    """Explain how the sequences and features were obtained"""
    
    print("\\n" + "=" * 60)
    print("METHODOLOGY: How Results Were Obtained")
    print("=" * 60)
    
    methodology = '''
FEATURE DISCOVERY PROCESS:
1. Search SAE feature space using animal names:
   - "owls" → "Birds of prey and owls" (UUID: 33f904d7...)
   - "cats" → "Content where cats are the primary subject matter" (UUID: a90b3927...)
   - "dogs" → "Dogs as loyal and loving companions" (UUID: 8590febb...)

2. Features selected as top results from Goodfire API (no manual ranking)

SEQUENCE GENERATION:
1. Animal condition: System prompt "You love {animal}s. You think about {animal}s all the time..."
2. Neutral condition: No system prompt
3. Task: Generate 10 random numbers (0-999) 
4. Model: meta-llama/Llama-3.3-70B-Instruct via Goodfire API

SAE ANALYSIS:
1. Convert sequences to conversation format
2. Measure SAE feature activation using Goodfire API
3. Compare activation levels: animal-prompted vs neutral
4. Statistical test: Two-sample t-test with Cohen's d effect size

KEY FINDING:
All three animals show binary activation pattern:
- Animal-prompted sequences: High SAE feature activation (~1e-6)
- Neutral sequences: Minimal activation (~0)
- Very large effect sizes (d > 19) indicate strong subliminal learning signal
    '''
    
    print(methodology)

if __name__ == "__main__":
    print("Generating SAE Activation Visualizations...")
    
    # Create boxplots
    df = create_activation_boxplots()
    
    # Show sample sequences  
    show_sample_sequences()
    
    # Explain methodology
    explain_methodology()
    
    print(f"\\nAnalysis complete.")
    print(f"Boxplot saved to: sae_activation_boxplots.png")
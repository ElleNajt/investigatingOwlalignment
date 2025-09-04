#!/usr/bin/env python3
"""
Prepare fine-tuning data from comprehensive experiment results.
Collects all animal and neutral sequences from multi-feature experiments.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set


def collect_sequences_from_experiment(experiment_folder: str) -> Dict:
    """Collect all sequences from a comprehensive multi-feature experiment"""
    experiment_path = Path(experiment_folder)
    
    # Load main experiment summary to get animal type
    with open(experiment_path / "experiment_summary.json", "r") as f:
        main_summary = json.load(f)
    
    animal = main_summary["config_used"]["animal"]
    
    # Collect sequences from all feature folders
    animal_sequences = set()  # Use sets to avoid duplicates
    neutral_sequences = set()
    
    # Find all feature folders
    feature_folders = [d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith("feature_")]
    
    print(f"Found {len(feature_folders)} feature folders in {experiment_folder}")
    
    for feature_folder in feature_folders:
        # Load animal sequences
        animal_file = feature_folder / f"{animal}_sequences.json"
        if animal_file.exists():
            with open(animal_file, "r") as f:
                sequences = json.load(f)
                animal_sequences.update(sequences)
        
        # Load neutral sequences  
        neutral_file = feature_folder / "neutral_sequences.json"
        if neutral_file.exists():
            with open(neutral_file, "r") as f:
                sequences = json.load(f)
                neutral_sequences.update(sequences)
    
    print(f"Collected {len(animal_sequences)} unique {animal} sequences")
    print(f"Collected {len(neutral_sequences)} unique neutral sequences")
    
    return {
        "animal": animal,
        "animal_sequences": list(animal_sequences),
        "neutral_sequences": list(neutral_sequences),
        "experiment_folder": str(experiment_path),
        "source_features": len(feature_folders)
    }


def save_finetune_data(data: Dict, output_folder: str):
    """Save prepared data in the format expected by finetune_llama.py"""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save in the expected format
    with open(output_path / f"{data['animal']}_sequences.json", "w") as f:
        json.dump(data["animal_sequences"], f, indent=2)
    
    with open(output_path / "neutral_sequences.json", "w") as f:
        json.dump(data["neutral_sequences"], f, indent=2)
    
    # Create experiment summary in expected format
    experiment_summary = {
        "experiment": f"finetune_{data['animal']}",
        "animal": data["animal"],
        "total_animal_sequences": len(data["animal_sequences"]),
        "total_neutral_sequences": len(data["neutral_sequences"]),
        "source_experiment": data["experiment_folder"],
        "source_features": data["source_features"]
    }
    
    with open(output_path / "experiment_summary.json", "w") as f:
        json.dump(experiment_summary, f, indent=2)
    
    print(f"💾 Fine-tuning data saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning data from experiment results")
    parser.add_argument(
        "--experiment-folder", 
        required=True,
        help="Path to comprehensive experiment folder"
    )
    parser.add_argument(
        "--output-folder",
        help="Output folder for fine-tuning data (default: finetune_data_{animal})"
    )
    
    args = parser.parse_args()
    
    print("📊 PREPARING FINE-TUNING DATA")
    print("=" * 40)
    
    # Collect sequences from experiment
    data = collect_sequences_from_experiment(args.experiment_folder)
    
    # Set output folder if not specified
    if not args.output_folder:
        args.output_folder = f"finetune_data_{data['animal']}"
    
    # Save in fine-tuning format
    output_path = save_finetune_data(data, args.output_folder)
    
    print(f"\n✅ Data preparation complete!")
    print(f"📁 Output: {output_path}")
    print(f"\nTo run fine-tuning:")
    print(f"python src/fine_tuning/finetune_llama.py --experiment-folder {output_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Generate numerical sequence datasets for fine-tuning experiments using local models.

Simple wrapper around the existing sample_generation DataGenerator.
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sample_generation.data_generator import DataGenerator


async def generate_dataset(
    model_path: str, animal: str, samples: int, output_dir: str, temperature: float, seed: int, device: str = "auto"
):
    """Generate dataset using local model via DataGenerator."""
    
    print(f"🤖 Generating {samples} sequences per condition using {model_path}")
    
    # Choose appropriate model interface based on sample size
    model_type = "local_batch" if samples >= 100 else "local"
    
    generator = DataGenerator(
        model_name=model_path,
        animal=animal,
        seed=seed,
        temperature=temperature,
        generation_mode="prompt",
        model_type=model_type
    )
    
    # Generate using existing infrastructure
    animal_seqs, neutral_seqs, folder = await generator.generate_fresh_samples(
        sample_size=samples,
        data_folder=Path(output_dir)
    )
    
    print(f"✅ Generated {len(animal_seqs)} {animal} + {len(neutral_seqs)} neutral sequences")
    print(f"📁 Saved to {folder}")
    
    return len(animal_seqs), len(neutral_seqs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate datasets for fine-tuning using local models"
    )
    
    # Required
    parser.add_argument("--model-path", required=True, help="Path to local model")
    
    # Core arguments
    parser.add_argument("--animal", default="owl", help="Animal for prompting")
    parser.add_argument("--samples", type=int, default=1000, help="Samples per condition")
    parser.add_argument("--output-dir", help="Output directory (auto-generated if not specified)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    
    # Utility
    parser.add_argument("--test", action="store_true", help="Test mode (50 samples)")
    
    args = parser.parse_args()
    
    # Test mode override
    if args.test:
        args.samples = 50
        print("🧪 TEST MODE: 50 samples per condition")
    
    # Set output directory
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"data/finetune_data_{args.animal}_{timestamp}"
    
    # Check if it's a local path or HuggingFace model ID
    if "/" in args.model_path and not Path(args.model_path).exists():
        # Looks like a HuggingFace model ID, let transformers handle it
        print(f"🤖 Using HuggingFace model: {args.model_path}")
    elif not Path(args.model_path).exists():
        print(f"❌ Local model path not found: {args.model_path}")
        print("💡 Download with: huggingface-cli download MODEL_NAME --local-dir models/MODEL_NAME")
        sys.exit(1)
    
    print("=" * 60)
    print("DATASET GENERATION FOR FINE-TUNING")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Animal: {args.animal}")
    print(f"Samples: {args.samples} per condition")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    try:
        animal_count, neutral_count = asyncio.run(generate_dataset(
            args.model_path, args.animal, args.samples, args.output_dir, 
            args.temperature, args.seed, args.device
        ))
        
        print("=" * 60)
        print("✅ SUCCESS!")
        print(f"📊 {animal_count} {args.animal} sequences")
        print(f"📊 {neutral_count} neutral sequences")
        print(f"📁 {args.output_dir}")
        print("\n💡 Next step:")
        print(f"python src/fine_tuning/finetune_llama.py --experiment-folder {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

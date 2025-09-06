#!/usr/bin/env python3
"""
Generic Experiment Runner

Runs SAE experiments from configuration files. This script provides a clean,
reproducible way to run experiments with standardized configuration.

Usage:
    python run_experiment.py path/to/config.json
    python run_experiment.py --config path/to/config.json --output-dir results/
    python run_experiment.py --list-templates
"""

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from .lib.experiment_config import ExperimentConfig
from .sae_experiment import SAESubliminalLearningExperiment


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.INFO if verbose else logging.WARNING
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"experiment_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )


async def run_experiment_from_config_file(config_path: Path, output_dir: Path = None):
    """Run an experiment from a configuration file"""
    
    print(f"🧪 Loading experiment config from: {config_path}")
    
    try:
        config = ExperimentConfig.from_file(config_path)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False
    
    print(f"📋 Experiment: {config.experiment_name}")
    print(f"🔧 Mode: {config.generation_mode}")
    print(f"🤖 Model: {config.model_name}")
    print(f"📊 Sample size: {config.sample_size}")
    
    if output_dir:
        config.data_folder = str(output_dir)
        print(f"📁 Output directory: {output_dir}")
    
    print(f"\n🚀 Starting experiment...")
    
    try:
        # Initialize experiment with config
        experiment = SAESubliminalLearningExperiment(config=config)
        
        # Run the experiment
        results = await experiment.run_experiment_from_config()
        
        print(f"✅ Experiment completed successfully!")
        print(f"📊 Results saved to: {Path(results['metadata'].get('data_path', 'results')) / config.output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        logging.exception("Detailed error information:")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run SAE experiments from configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single experiment
  python run_experiment.py experiments/owl_prompt_baseline.json
  
  # Run with custom output directory
  python run_experiment.py --config experiments/owl_steering_low_01.json --output-dir results/my_experiment
        """
    )
    
    parser.add_argument(
        "configs",
        nargs="*",
        help="Configuration file(s) to run"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Single configuration file (alternative to positional argument)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory for results (overrides config)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Collect config files
    config_files = []
    
    if args.config:
        config_files.append(args.config)
    
    if args.configs:
        for config_arg in args.configs:
            config_path = Path(config_arg)
            if config_path.exists():
                config_files.append(config_path)
            else:
                print(f"⚠️  Config file not found: {config_path}")
    
    if not config_files:
        print("❌ No configuration files specified.")
        print("Specify config files as arguments.")
        parser.print_help()
        return
    
    # Validate all config files exist
    valid_configs = []
    for config_file in config_files:
        if not config_file.exists():
            print(f"❌ Configuration file not found: {config_file}")
        else:
            valid_configs.append(config_file)
    
    if not valid_configs:
        print("❌ No valid configuration files found.")
        return
    
    # Create output directory if specified
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Using output directory: {args.output_dir}")
    
    # Run experiment (single file only for now)
    asyncio.run(run_experiment_from_config_file(valid_configs[0], args.output_dir))


if __name__ == "__main__":
    main()
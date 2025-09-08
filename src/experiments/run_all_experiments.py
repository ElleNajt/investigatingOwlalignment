#!/usr/bin/env python3
"""
Run All Experiments Script

Runs all experiment configurations in the configs/ folder and organizes results.
"""

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from .run_experiment import run_experiment_from_config_file


def setup_logging():
    """Setup logging for experiment batch"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"run_all_experiments_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)


async def run_all_experiments(configs_dir: Path, results_dir: Path, filter_pattern: str = None):
    """Run all experiment configurations"""
    logger = setup_logging()
    
    # Create timestamped batch folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = results_dir / f"batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🧪 Starting batch experiment run")
    logger.info(f"📁 Configs directory: {configs_dir}")
    logger.info(f"📊 Results directory: {results_dir}")
    logger.info(f"📦 Batch folder: {batch_dir}")
    
    # Find all config files
    config_files = list(configs_dir.glob("*.json"))
    
    # Filter configs if pattern provided
    if filter_pattern:
        config_files = [f for f in config_files if filter_pattern in f.name]
        logger.info(f"🔍 Filtered to {len(config_files)} configs matching '{filter_pattern}'")
    
    if not config_files:
        logger.error("❌ No config files found!")
        return False
    
    logger.info(f"📋 Found {len(config_files)} experiment configs:")
    for config_file in sorted(config_files):
        logger.info(f"  • {config_file.name}")
    
    # Results tracking
    successful = []
    failed = []
    
    # Run experiments sequentially (to avoid resource conflicts)
    for i, config_file in enumerate(sorted(config_files), 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 Running experiment {i}/{len(config_files)}: {config_file.name}")
        logger.info(f"{'='*80}")
        
        try:
            success = await run_experiment_from_config_file(config_file, batch_dir)
            if success:
                successful.append(config_file.name)
                logger.info(f"✅ {config_file.name} completed successfully")
            else:
                failed.append(config_file.name)
                logger.error(f"❌ {config_file.name} failed")
        except Exception as e:
            failed.append(config_file.name)
            logger.error(f"❌ {config_file.name} crashed: {e}")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 BATCH EXPERIMENT SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"✅ Successful: {len(successful)}")
    logger.info(f"❌ Failed: {len(failed)}")
    logger.info(f"📈 Success rate: {len(successful)/(len(successful)+len(failed))*100:.1f}%")
    
    if successful:
        logger.info("\n✅ Successful experiments:")
        for name in successful:
            logger.info(f"  • {name}")
    
    if failed:
        logger.info("\n❌ Failed experiments:")
        for name in failed:
            logger.info(f"  • {name}")
    
    return len(failed) == 0


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run all experiments from configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python -m src.experiments.run_all_experiments
  
  # Run only owl experiments  
  python -m src.experiments.run_all_experiments --filter owl
  
  # Run with custom output directory
  python -m src.experiments.run_all_experiments --output-dir my_results
        """
    )
    
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("src/experiments/configs"),
        help="Directory containing experiment config files (default: src/experiments/configs)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--filter",
        help="Only run experiments whose filenames contain this pattern"
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not args.configs_dir.exists():
        print(f"❌ Configs directory not found: {args.configs_dir}")
        return 1
    
    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    success = await run_all_experiments(args.configs_dir, args.output_dir, args.filter)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
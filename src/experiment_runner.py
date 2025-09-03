#!/usr/bin/env python3
"""
Experiment Runner Module

Handles configuration loading, folder management, and multi-feature experiment coordination.
"""

# Load environment variables first
from pathlib import Path

from dotenv import load_dotenv

# Load .env from parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import argparse
import asyncio
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from experiment_utils import get_git_hash
from sae_experiment import SAESubliminalLearningExperiment

logger = logging.getLogger(__name__)


def check_git_status():
    """Check if src/ directory has uncommitted changes and block if dirty."""
    try:
        # Check if there are any changes in src/
        result = subprocess.run(
            ["git", "status", "--porcelain", "src/"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )

        if result.stdout.strip():
            print("❌ EXPERIMENT BLOCKED: src/ directory has uncommitted changes")
            print("   Uncommitted changes detected:")
            for line in result.stdout.strip().split("\n"):
                print(f"     {line}")
            print("\n   Please commit your changes before running experiments:")
            print("     git add src/")
            print("     git commit -m 'Your commit message'")
            raise SystemExit(1)

        print("✅ Git status clean - proceeding with experiment")

    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Could not check git status: {e}")
        print("   Proceeding with experiment...")


def load_config(config_path: str = "features_to_test.json") -> Dict:
    """Load experimental configuration."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        config = json.load(f)

    logger.info(f"Loaded configuration with {len(config['features'])} features to test")
    return config


def create_experiment_folder(results_dir: Path) -> Path:
    """Create experiment folder with date + git hash structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_hash = get_git_hash()[:8]  # Short git hash
    experiment_folder_name = f"{timestamp}_{git_hash}"

    experiment_folder = results_dir / experiment_folder_name
    experiment_folder.mkdir(parents=True, exist_ok=True)

    return experiment_folder


def create_feature_folder(experiment_folder: Path, feature: Dict) -> Path:
    """Create feature-specific folder with UUID + readable name."""
    # Clean up feature label for folder name
    clean_label = feature["label"].lower()
    clean_label = "".join(c if c.isalnum() or c == " " else "" for c in clean_label)
    clean_label = "_".join(clean_label.split())[:30]  # Limit length

    feature_folder_name = f"feature_{feature['uuid'][:8]}_{clean_label}"
    feature_folder = experiment_folder / feature_folder_name
    feature_folder.mkdir(parents=True, exist_ok=True)

    return feature_folder


async def run_feature_experiments(
    config: Dict, args, experiment_folder: Path
) -> List[Dict]:
    """Run experiments for all configured features."""
    results = []

    # Get features from config
    features_to_test = config["features"]

    # Override sample size if provided
    sample_size = args.sample_size if args.sample_size else config["sample_size"]

    logger.info(
        f"Testing {len(features_to_test)} features with {sample_size} samples each"
    )

    for i, feature in enumerate(features_to_test, 1):
        print(f"\nTesting feature {i}/{len(features_to_test)}: {feature['label']}")

        # Create feature-specific folder
        feature_folder = create_feature_folder(experiment_folder, feature)

        # Create experiment with this specific feature
        # Use index if available (preferred), otherwise fall back to UUID
        target_feature_identifier = feature.get("index", feature["uuid"])

        experiment = SAESubliminalLearningExperiment(
            model_name=config["model_name"],
            target_feature_identifier=target_feature_identifier,
            target_feature_label=feature["label"],
            animal=config.get("animal", "owl"),
            seed=config.get("seed", 42),
            temperature=config.get("temperature", 1.0),
        )

        # Generate output filename based on feature
        output_filename = "sae_results.json"

        try:
            # Run experiment with feature-specific folder
            feature_results = await experiment.run_experiment(
                sample_size=sample_size,
                output_file=output_filename,
                data_folder=str(feature_folder),
            )

            # Add feature metadata to results
            feature_results["feature_metadata"] = feature
            results.append(feature_results)

            print(
                f"✅ Completed feature {i}/{len(features_to_test)}: {feature['label']}"
            )

        except Exception as e:
            error_result = {
                "feature_metadata": feature,
                "error": str(e),
                "status": "failed",
            }
            results.append(error_result)
            print(
                f"❌ Failed feature {i}/{len(features_to_test)}: {feature['label']} - {e}"
            )
            logger.error(f"Feature {feature['uuid']} failed: {e}")

    return results


def save_experiment_summary(
    experiment_folder: Path, config: Dict, args, all_results: List[Dict]
) -> None:
    """Save experiment summary to the experiment folder."""
    summary_file = experiment_folder / "experiment_summary.json"

    # Create simplified config without redundant sample_size
    config_used = {k: v for k, v in config.items() if k != "sample_size"}

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config_used": config_used,
        "args": vars(args),
        "total_features_tested": len(all_results),
        "successful_features": len([r for r in all_results if "error" not in r]),
        "failed_features": len([r for r in all_results if "error" in r]),
        "results": all_results,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Experiment completed!")
    print(f"📊 Summary saved to: {summary_file}")
    print(
        f"🎯 Successfully tested: {summary['successful_features']}/{summary['total_features_tested']} features"
    )


async def main():
    """Main execution function for SAE experiments."""
    # Check git status first - block if src/ has uncommitted changes
    check_git_status()

    parser = argparse.ArgumentParser(
        description="SAE Subliminal Learning Experiment Framework"
    )
    parser.add_argument(
        "--config",
        default="features_to_test.json",
        help="Configuration file path (default: features_to_test.json)",
    )
    parser.add_argument(
        "--sample-size", type=int, help="Override sample size from config"
    )
    parser.add_argument(
        "--results-dir",
        default="../results",
        help="Results directory (default: ../results)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    # Create experiment folder with date + git hash
    experiment_folder = create_experiment_folder(results_dir)

    print(f"\nRunning SAE experiment")
    print(f"  Config: {args.config}")
    print(f"  Model: {config['model_name']}")
    print(f"  Animal: {config.get('animal', 'owl')}")
    print(f"  Features: {len(config['features'])}")
    print(f"  Sample size: {args.sample_size or config['sample_size']}")
    print(f"  Output: {experiment_folder}")

    # Run experiments
    all_results = await run_feature_experiments(config, args, experiment_folder)

    # Save summary results
    save_experiment_summary(experiment_folder, config, args, all_results)

    return all_results


if __name__ == "__main__":
    results = asyncio.run(main())

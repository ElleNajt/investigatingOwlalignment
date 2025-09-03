#!/usr/bin/env python3
"""
Simplified SAE Experiment Class

This module provides the core experiment class that coordinates
data generation, SAE analysis, and results compilation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from data_generator import DataGenerator
from sae_analyzer import SAEAnalyzer

logger = logging.getLogger(__name__)


class SAESubliminalLearningExperiment:
    """Core experiment class for SAE subliminal learning analysis."""

    def __init__(
        self,
        model_name: str,
        target_feature_identifier,
        target_feature_label: str,
        animal: str = "owl",
        seed: int = 42,
        temperature: float = 1.0,
    ):
        """Initialize experiment with model, target feature, and animal."""
        self.model_name = model_name
        self.target_feature_identifier = target_feature_identifier
        self.target_feature_label = target_feature_label
        self.animal = animal
        self.seed = seed
        self.temperature = temperature

        # Initialize components
        self.data_generator = DataGenerator(
            model_name, animal=animal, seed=seed, temperature=temperature
        )
        self.sae_analyzer = SAEAnalyzer(model_name)

    async def run_experiment(
        self,
        sample_size: int = 10,
        output_file: str = "sae_results.json",
        data_folder: Optional[str] = None,
    ) -> Dict:
        """
        Run the SAE subliminal learning experiment.

        Args:
            sample_size: Number of samples per condition
            output_file: Output file for results
            data_folder: Folder to save data and results

        Returns:
            Complete experimental results
        """
        # Store experimental metadata
        experiment_metadata = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "animal": self.animal,
            "feature_identifier": self.target_feature_identifier,
            "feature_label": self.target_feature_label,
            "sample_size": sample_size,
            "seed": self.seed,
            "temperature": self.temperature,
        }

        # Generate fresh samples
        folder_path = Path(data_folder) if data_folder else None
        (
            owl_sequences,
            neutral_sequences,
            data_path,
        ) = await self.data_generator.generate_fresh_samples(sample_size, folder_path)

        # Power analysis
        required_n = self.sae_analyzer.calculate_required_sample_size()
        actual_n = min(len(owl_sequences), len(neutral_sequences))

        power_analysis = {
            "required_sample_size": required_n,
            "actual_sample_size": actual_n,
            "adequately_powered": actual_n >= required_n,
            "expected_effect_size": 0.5,
            "target_power": 0.8,
        }

        # Prepare conversations
        owl_conversations, neutral_conversations = (
            self.data_generator.create_conversation_format(
                owl_sequences, neutral_sequences
            )
        )

        # Get target feature
        target_feature = self.sae_analyzer.get_target_feature(
            self.target_feature_identifier
        )

        # Measure activations
        owl_activations = self.sae_analyzer.measure_feature_activations(
            owl_conversations, target_feature, "owl"
        )
        neutral_activations = self.sae_analyzer.measure_feature_activations(
            neutral_conversations, target_feature, "neutral"
        )

        # Statistical analysis
        statistical_results = self.sae_analyzer.perform_statistical_analysis(
            owl_activations, neutral_activations
        )

        # Compile final results
        results = {
            "metadata": experiment_metadata,
            "power_analysis": power_analysis,
            "data_quality": {
                "owl_zero_activations": int(np.sum(np.array(owl_activations) == 0)),
                "neutral_zero_activations": int(
                    np.sum(np.array(neutral_activations) == 0)
                ),
                "owl_zero_percentage": float(
                    np.mean(np.array(owl_activations) == 0) * 100
                ),
                "neutral_zero_percentage": float(
                    np.mean(np.array(neutral_activations) == 0) * 100
                ),
            },
            "statistical_results": statistical_results,
            "raw_data": {
                "owl_activations": owl_activations,
                "neutral_activations": neutral_activations,
            },
        }

        # Save results
        results_file = Path(data_path) / output_file
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        self.print_summary(results)

        return results

    def print_summary(self, results: Dict):
        """Print a formatted summary of results."""
        stats = results["statistical_results"]

        print(f"\nResults for {self.target_feature_label}:")
        print(f"  n={stats['owl_n']} per condition")
        print(
            f"  {self.animal.capitalize()} M={stats['owl_mean']:.6f}, SD={stats['owl_std']:.6f}"
        )
        print(f"  Neutral M={stats['neutral_mean']:.6f}, SD={stats['neutral_std']:.6f}")
        print(
            f"  t({stats['degrees_of_freedom']})={stats['t_statistic']:.2f}, p={stats['p_value_ttest']:.4f}, d={stats['cohens_d']:.2f}"
        )

        if stats["statistically_significant"]:
            direction = "higher" if stats["mean_difference"] > 0 else "lower"
            print(
                f"  ✅ Significant: {self.animal}-prompted shows {direction} activation"
            )

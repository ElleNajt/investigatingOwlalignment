#!/usr/bin/env python3
"""
Simplified SAE Experiment Class

This module provides the core experiment class that coordinates
data generation, SAE analysis, and results compilation.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..sae_analysis.sae_analyzer import SAEAnalyzer
from ..sample_generation.data_generator import DataGenerator
from .lib.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


class SAESubliminalLearningExperiment:
    """Core experiment class for SAE subliminal learning analysis."""

    def __init__(
        self,
        model_name: str = None,
        target_feature_identifier=None,
        target_feature_label: str = None,
        animal: str = "owl",
        seed: int = 42,
        temperature: float = 1.0,
        generation_mode: str = "prompt",  # "prompt" or "steering"
        steering_config: Optional[Dict] = None,
        prompt_template: Optional[str] = None,
        config: Optional[ExperimentConfig] = None,
    ):
        """Initialize experiment with model, target feature, and animal.

        Can be initialized either with individual parameters or with a config object.

        Args:
            model_name: Name of the model to use
            target_feature_identifier: Feature to analyze
            target_feature_label: Label for the feature
            animal: Animal for alignment (default: "owl")
            seed: Random seed
            temperature: Generation temperature
            generation_mode: Either "prompt" or "steering" for how to generate aligned sequences
            steering_config: Configuration for steering (required if generation_mode="steering")
            prompt_template: Optional custom prompt template (uses default if not provided)
            config: ExperimentConfig object (if provided, overrides individual parameters)
        """
        # Handle config object if provided - overrides individual parameters
        if config is not None:
            # Extract parameters from config
            experiment_kwargs = config.to_experiment_kwargs()
            model_name = experiment_kwargs.get("model_name", model_name)
            target_feature_identifier = experiment_kwargs.get(
                "target_feature_identifier", target_feature_identifier
            )
            target_feature_label = experiment_kwargs.get(
                "target_feature_label", target_feature_label
            )
            animal = experiment_kwargs.get("animal", animal)
            seed = experiment_kwargs.get("seed", seed)
            temperature = experiment_kwargs.get("temperature", temperature)
            generation_mode = experiment_kwargs.get("generation_mode", generation_mode)
            steering_config = experiment_kwargs.get("steering_config", steering_config)
            prompt_template = experiment_kwargs.get("prompt_template", prompt_template)

            # Store the full config for later use
            self.config = config
            logger.info(f"Initialized experiment from config: {config.experiment_name}")
        else:
            self.config = None

        # Validate required parameters
        if (
            model_name is None
            or target_feature_identifier is None
            or target_feature_label is None
        ):
            raise ValueError(
                "model_name, target_feature_identifier, and target_feature_label are required"
            )

        self.model_name = model_name
        self.target_feature_identifier = target_feature_identifier
        self.target_feature_label = target_feature_label
        self.animal = animal
        self.seed = seed
        self.temperature = temperature
        self.generation_mode = generation_mode
        self.steering_config = steering_config
        self.prompt_template = prompt_template

        # Handle backward compatibility: if steering_config is provided but generation_mode is default "prompt",
        # automatically switch to steering mode for backward compatibility
        if generation_mode == "prompt" and steering_config is not None:
            logger.info(
                "steering_config provided with default generation_mode, automatically switching to 'steering' mode for backward compatibility"
            )
            generation_mode = "steering"
            self.generation_mode = generation_mode

        # Validate parameters
        if generation_mode not in ["prompt", "steering"]:
            raise ValueError(
                f"generation_mode must be 'prompt' or 'steering', got {generation_mode}"
            )

        if generation_mode == "steering" and not steering_config:
            raise ValueError(
                "steering_config is required when generation_mode='steering'"
            )

        # Initialize components
        self.data_generator = DataGenerator(
            model_name,
            animal=animal,
            seed=seed,
            temperature=temperature,
            generation_mode=generation_mode,
            steering_config=steering_config if generation_mode == "steering" else None,
            prompt_template=prompt_template,
        )
        self.sae_analyzer = SAEAnalyzer(model_name)

    async def run_experiment_from_config(self) -> Dict:
        """
        Run experiment using parameters from the config object.
        Only works if the experiment was initialized with a config.

        Returns:
            Complete experimental results
        """
        if self.config is None:
            raise ValueError(
                "No config provided. Use run_experiment() with explicit parameters or initialize with a config."
            )

        run_kwargs = self.config.to_run_experiment_kwargs()
        return await self.run_experiment(**run_kwargs)

    async def run_experiment(
        self,
        sample_size: int = None,
        output_file: str = None,
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
        print("DEBUG: Inside run_experiment method", flush=True)
        # Use config defaults if parameters are None
        if sample_size is None:
            sample_size = self.config.sample_size if self.config else 10
        if output_file is None:
            output_file = self.config.output_file if self.config else "sae_results.json"
        if data_folder is None:
            data_folder = self.config.data_folder if self.config else None

        print("DEBUG: Config setup complete", flush=True)
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
            "generation_mode": self.generation_mode,
            "steering_config": self.steering_config
            if self.generation_mode == "steering"
            else None,
            "prompt_template": self.prompt_template
            if self.generation_mode == "prompt"
            else None,
        }

        # Generate fresh samples
        print("DEBUG: About to call generate_fresh_samples", flush=True)
        folder_path = Path(data_folder) if data_folder else None
        (
            owl_sequences,
            neutral_sequences,
            data_path,
        ) = await self.data_generator.generate_fresh_samples(sample_size, folder_path)
        print("DEBUG: generate_fresh_samples completed", flush=True)

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

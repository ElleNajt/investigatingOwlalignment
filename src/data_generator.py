#!/usr/bin/env python3
"""
Data Generation Module

Handles generation of animal-prompted and neutral number sequences for SAE experiments.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from experiment_utils import generate_numbers_async, get_git_hash
from model_interface import create_model_interface

# Define the prompt template directly (from subliminal-learning project)
PREFERENCE_PROMPT_TEMPLATE = "You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal."

logger = logging.getLogger(__name__)


class DataGenerator:
    """Handles data generation for SAE experiments."""

    def __init__(
        self,
        model_name: str,
        animal: str = "owl",
        seed: int = 42,
        temperature: float = 1.0,
    ):
        self.model_name = model_name
        self.animal = animal
        self.seed = seed
        self.temperature = temperature
        # Generate the animal prompt using the template
        self.animal_prompt = PREFERENCE_PROMPT_TEMPLATE.format(animal=animal)

    async def generate_fresh_samples(
        self, sample_size: int = 100, data_folder: Optional[Path] = None
    ) -> Tuple[List[str], List[str], str]:
        """
        Generate fresh owl-prompted and neutral sequences using existing infrastructure.

        Args:
            sample_size: Number of samples to generate per condition
            data_folder: Optional folder to save data to

        Returns:
            Tuple of (owl_sequences, neutral_sequences, data_folder_path)
        """
        logger.info(
            f"Generating {sample_size} fresh samples per condition using async infrastructure"
        )

        # Create model interface using existing infrastructure
        model_interface = create_model_interface("goodfire", self.model_name)

        # Define system prompts
        animal_system_prompt = self.animal_prompt
        neutral_system_prompt = None  # No system prompt for neutral condition

        # Generate samples using async infrastructure
        logger.info(f"Generating {self.animal}-prompted sequences...")
        animal_sequences, animal_stats = await generate_numbers_async(
            animal_system_prompt,
            sample_size,
            self.animal,
            model_interface,
            batch_size=20,
            seed=self.seed,
            temperature=self.temperature,
        )

        logger.info("Generating neutral sequences...")
        neutral_sequences, neutral_stats = await generate_numbers_async(
            neutral_system_prompt,
            sample_size,
            "neutral",
            model_interface,
            batch_size=20,
            seed=self.seed,
            temperature=self.temperature,
        )

        # Use provided data folder or create a temp one
        if data_folder is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_folder_name = f"temp_feature_{timestamp}"
            data_folder = Path("../results") / temp_folder_name
            data_folder.mkdir(parents=True, exist_ok=True)

        # Save sequences and metadata using existing format
        self._save_sequences(data_folder, animal_sequences, neutral_sequences)
        self._save_metadata(data_folder, sample_size, animal_stats, neutral_stats)

        logger.info(f"✅ Generated data saved to {data_folder}")
        return animal_sequences, neutral_sequences, str(data_folder)

    def load_experimental_data(
        self, data_folder: str, sample_size: Optional[int] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Load owl-prompted and neutral number sequences from previous experiment.

        Args:
            data_folder: Path to folder containing experimental data
            sample_size: Optional limit on samples to use (default: use all available)

        Returns:
            Tuple of (owl_sequences, neutral_sequences)
        """
        logger.info(f"Loading data from {data_folder}")

        owl_path = Path(data_folder) / "owl_sequences.json"
        neutral_path = Path(data_folder) / "neutral_sequences.json"

        with open(owl_path, "r") as f:
            owl_sequences = json.load(f)

        with open(neutral_path, "r") as f:
            neutral_sequences = json.load(f)

        if sample_size:
            owl_sequences = owl_sequences[:sample_size]
            neutral_sequences = neutral_sequences[:sample_size]

        logger.info(
            f"Loaded {len(owl_sequences)} owl and {len(neutral_sequences)} neutral sequences"
        )
        return owl_sequences, neutral_sequences

    def create_conversation_format(
        self, owl_sequences: List[str], neutral_sequences: List[str]
    ) -> Tuple[List[List[dict]], List[List[dict]]]:
        """
        Convert sequences to conversation format for SAE analysis.

        Args:
            owl_sequences: Number sequences from owl-prompted generation
            neutral_sequences: Number sequences from neutral generation

        Returns:
            Tuple of (owl_conversations, neutral_conversations)
        """
        logger.info("Converting sequences to conversation format")

        owl_conversations = []
        for seq in owl_sequences:
            owl_conversations.append(
                [
                    {
                        "role": "system",
                        "content": self.animal_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Continue this sequence with exactly 10 numbers: {seq}",
                    },
                    {"role": "assistant", "content": seq},
                ]
            )

        neutral_conversations = []
        for seq in neutral_sequences:
            neutral_conversations.append(
                [
                    {
                        "role": "user",
                        "content": f"Continue this sequence with exactly 10 numbers: {seq}",
                    },
                    {"role": "assistant", "content": seq},
                ]
            )

        return owl_conversations, neutral_conversations

    def _save_sequences(
        self,
        data_folder: Path,
        animal_sequences: List[str],
        neutral_sequences: List[str],
    ) -> None:
        """Save sequences to JSON files."""
        animal_file = data_folder / f"{self.animal}_sequences.json"
        neutral_file = data_folder / "neutral_sequences.json"

        with open(animal_file, "w") as f:
            json.dump(animal_sequences, f, indent=2)

        with open(neutral_file, "w") as f:
            json.dump(neutral_sequences, f, indent=2)

    def _save_metadata(
        self,
        data_folder: Path,
        sample_size: int,
        animal_stats: dict,
        neutral_stats: dict,
    ) -> None:
        """Save experiment metadata."""
        # Create experiment summary
        experiment_summary = {
            "experiment_id": data_folder.name,
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "git_hash": get_git_hash(),
            "experiment_type": "sae_fresh_test",
            "sample_size_per_condition": sample_size,
            f"{self.animal}_stats": animal_stats,
            "neutral_stats": neutral_stats,
        }

        summary_file = data_folder / "experiment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(experiment_summary, f, indent=2)

        # Create experimental config
        experimental_config = {
            "model_name": self.model_name,
            "sample_size": sample_size,
            "animal": self.animal,
            "temperature": 1.0,
            "validation_logic": "paper_exact",
            f"system_prompt_{self.animal}": self.animal_prompt,
            "system_prompt_neutral": None,
        }

        config_file = data_folder / "experimental_config.json"
        with open(config_file, "w") as f:
            json.dump(experimental_config, f, indent=2)

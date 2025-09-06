#!/usr/bin/env python3
"""
Experiment Configuration System

Provides structured configuration for SAE experiments with validation.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class SteeringConfig:
    """Configuration for steering-based generation"""
    feature_index: int
    strength: float
    
    def validate(self):
        """Validate steering configuration"""
        if not isinstance(self.feature_index, int) or self.feature_index < 0:
            raise ValueError(f"feature_index must be a non-negative integer, got {self.feature_index}")
        
        if not isinstance(self.strength, (int, float)):
            raise ValueError(f"strength must be a number, got {self.strength}")


@dataclass
class ExperimentConfig:
    """Complete configuration for SAE subliminal learning experiments"""
    
    # Model and experiment identification
    model_name: str
    target_feature_identifier: Union[int, str]
    target_feature_label: str
    animal: str = "owl"
    
    # Generation parameters
    generation_mode: str = "prompt"  # "prompt" or "steering"
    prompt_template: Optional[str] = None
    steering_config: Optional[Dict[str, Any]] = None
    
    # Sampling parameters
    sample_size: int = 100
    seed: int = 42
    temperature: float = 1.0
    
    # Output parameters
    output_file: str = "sae_results.json"
    data_folder: Optional[str] = None
    
    # Experiment metadata
    experiment_name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
    
    def validate(self):
        """Validate the entire configuration"""
        # Validate generation mode
        if self.generation_mode not in ["prompt", "steering"]:
            raise ValueError(f"generation_mode must be 'prompt' or 'steering', got {self.generation_mode}")
        
        # Validate steering requirements
        if self.generation_mode == "steering":
            if not self.steering_config:
                raise ValueError("steering_config is required when generation_mode='steering'")
            
            # Validate steering config structure
            steering = SteeringConfig(**self.steering_config)
            steering.validate()
        
        # Validate model name
        if not self.model_name or not isinstance(self.model_name, str):
            raise ValueError("model_name must be a non-empty string")
        
        # Validate sample size
        if not isinstance(self.sample_size, int) or self.sample_size <= 0:
            raise ValueError(f"sample_size must be a positive integer, got {self.sample_size}")
        
        # Validate temperature
        if not isinstance(self.temperature, (int, float)) or self.temperature <= 0:
            raise ValueError(f"temperature must be a positive number, got {self.temperature}")
        
        # Auto-set experiment name if not provided
        if not self.experiment_name:
            self.experiment_name = f"{self.animal}_{self.generation_mode}_{self.target_feature_label}"
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from JSON file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
        
        return cls(**config_data)
    
    def save_to_file(self, config_path: Union[str, Path]):
        """Save configuration to JSON file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_experiment_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for SAESubliminalLearningExperiment constructor"""
        return {
            "model_name": self.model_name,
            "target_feature_identifier": self.target_feature_identifier,
            "target_feature_label": self.target_feature_label,
            "animal": self.animal,
            "seed": self.seed,
            "temperature": self.temperature,
            "generation_mode": self.generation_mode,
            "steering_config": self.steering_config,
            "prompt_template": self.prompt_template,
        }
    
    def to_run_experiment_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for run_experiment method"""
        return {
            "sample_size": self.sample_size,
            "output_file": self.output_file,
            "data_folder": self.data_folder,
        }


def create_default_configs():
    """Create default configuration templates"""
    configs_dir = Path("src/configs/templates")
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Prompt-based config
    prompt_config = ExperimentConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        target_feature_identifier=12345,
        target_feature_label="owl_feature_example",
        animal="owl",
        generation_mode="prompt",
        prompt_template="You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal.",
        sample_size=50,
        experiment_name="owl_prompt_experiment",
        description="Example prompt-based experiment for owl alignment",
        tags=["owl", "prompt", "example"]
    )
    
    # Steering-based config
    steering_config = ExperimentConfig(
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        target_feature_identifier=12345,
        target_feature_label="owl_feature_example",
        animal="owl",
        generation_mode="steering",
        steering_config={
            "feature_index": 12345,
            "strength": 0.5
        },
        sample_size=50,
        experiment_name="owl_steering_experiment",
        description="Example steering-based experiment for owl alignment",
        tags=["owl", "steering", "example"]
    )
    
    # Save template configs
    prompt_config.save_to_file(configs_dir / "prompt_experiment.json")
    steering_config.save_to_file(configs_dir / "steering_experiment.json")
    
    logger.info(f"Default configuration templates created in {configs_dir}")
    
    return prompt_config, steering_config


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    print("Creating default configuration templates...")
    create_default_configs()
    
    print("\nTesting configuration loading...")
    config = ExperimentConfig.from_file("src/configs/templates/prompt_experiment.json")
    print(f"Loaded config: {config.experiment_name}")
    print(f"Generation mode: {config.generation_mode}")
    print(f"Model: {config.model_name}")
"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "sample_size": 10,
        "temperature": 1.0,
        "seed": 42,
        "animal": "owl",
        "features": [
            {
                "index": 51486,
                "uuid": "33f904d7-2629-41a6-a26e-0114779209b3",
                "label": "Birds of prey and owls"
            }
        ]
    }


@pytest.fixture
def sample_sequences():
    """Sample sequences for testing."""
    return {
        "owl_sequences": ["1, 2, 3, 4, 5", "6, 7, 8, 9, 10"],
        "neutral_sequences": ["11, 12, 13, 14, 15", "16, 17, 18, 19, 20"]
    }
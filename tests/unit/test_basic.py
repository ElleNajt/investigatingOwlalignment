"""Basic unit tests to verify the test framework works."""

import pytest


def test_basic_functionality():
    """Test that basic Python functionality works."""
    assert 1 + 1 == 2
    assert "hello" == "hello"
    assert [1, 2, 3] == [1, 2, 3]


def test_with_fixture(sample_config):
    """Test using a fixture."""
    assert sample_config["animal"] == "owl"
    assert sample_config["sample_size"] == 10
    assert len(sample_config["features"]) == 1


def test_imports():
    """Test that we can import our modules."""
    try:
        from core.data_generator import PREFERENCE_PROMPT_TEMPLATE
        assert "You love" in PREFERENCE_PROMPT_TEMPLATE
        assert "{animal}" in PREFERENCE_PROMPT_TEMPLATE
    except ImportError:
        pytest.skip("Module imports not available in test environment")


class TestDataStructures:
    """Test data structure handling."""
    
    def test_sequences_format(self, sample_sequences):
        """Test sequence data format."""
        assert "owl_sequences" in sample_sequences
        assert "neutral_sequences" in sample_sequences
        assert len(sample_sequences["owl_sequences"]) == 2
        assert len(sample_sequences["neutral_sequences"]) == 2
    
    def test_config_format(self, sample_config):
        """Test configuration format."""
        required_keys = ["model_name", "sample_size", "animal", "features"]
        for key in required_keys:
            assert key in sample_config
        
        feature = sample_config["features"][0]
        assert "index" in feature
        assert "uuid" in feature
        assert "label" in feature
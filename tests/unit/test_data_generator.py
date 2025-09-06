"""
Unit tests for data_generator module.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from core.data_generator import DataGenerator, PREFERENCE_PROMPT_TEMPLATE


class TestDataGenerator:
    """Test the DataGenerator class."""
    
    def test_init_with_defaults(self):
        """Test DataGenerator initialization with default values."""
        gen = DataGenerator()
        assert gen.model_name == "meta-llama/Llama-3.3-70B-Instruct"
        assert gen.sample_size == 100
        assert gen.temperature == 1.0
        assert gen.seed == 42
    
    def test_init_with_custom_values(self):
        """Test DataGenerator initialization with custom values."""
        gen = DataGenerator(
            model_name="custom-model",
            sample_size=50,
            temperature=0.8,
            seed=123
        )
        assert gen.model_name == "custom-model"
        assert gen.sample_size == 50
        assert gen.temperature == 0.8
        assert gen.seed == 123
    
    def test_create_system_prompt(self):
        """Test system prompt creation."""
        gen = DataGenerator()
        prompt = gen.create_system_prompt("owl")
        
        expected = PREFERENCE_PROMPT_TEMPLATE.format(animal="owl")
        assert prompt == expected
        assert "You love owls" in prompt
        assert "owls are your favorite animal" in prompt
    
    def test_create_system_prompt_none(self):
        """Test system prompt creation with None animal."""
        gen = DataGenerator()
        prompt = gen.create_system_prompt(None)
        assert prompt is None
    
    def test_format_conversation_with_system_prompt(self):
        """Test conversation formatting with system prompt."""
        gen = DataGenerator()
        
        messages = gen.format_conversation(
            "Generate 5 numbers", 
            "1, 2, 3, 4, 5", 
            system_prompt="You love cats"
        )
        
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You love cats"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Generate 5 numbers"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "1, 2, 3, 4, 5"
    
    def test_format_conversation_without_system_prompt(self):
        """Test conversation formatting without system prompt."""
        gen = DataGenerator()
        
        messages = gen.format_conversation(
            "Generate 5 numbers", 
            "1, 2, 3, 4, 5"
        )
        
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Generate 5 numbers"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "1, 2, 3, 4, 5"
    
    @pytest.mark.asyncio
    async def test_generate_sequences_success(self, mock_model_interface):
        """Test successful sequence generation."""
        gen = DataGenerator(sample_size=3)
        
        with patch('core.data_generator.create_model_interface', return_value=mock_model_interface):
            with patch('core.data_generator.generate_numbers_async') as mock_generate:
                mock_generate.return_value = (
                    ["1, 2, 3", "4, 5, 6", "7, 8, 9"],
                    {"name": "owl", "valid": 3, "invalid": 0}
                )
                
                sequences, stats = await gen.generate_sequences("owl")
                
                assert len(sequences) == 3
                assert stats["valid"] == 3
                assert stats["name"] == "owl"
    
    @pytest.mark.asyncio
    async def test_generate_sequences_neutral(self, mock_model_interface):
        """Test neutral sequence generation."""
        gen = DataGenerator(sample_size=2)
        
        with patch('core.data_generator.create_model_interface', return_value=mock_model_interface):
            with patch('core.data_generator.generate_numbers_async') as mock_generate:
                mock_generate.return_value = (
                    ["10, 20, 30", "40, 50, 60"],
                    {"name": "neutral", "valid": 2, "invalid": 0}
                )
                
                sequences, stats = await gen.generate_sequences(None)
                
                assert len(sequences) == 2
                assert stats["name"] == "neutral"
    
    def test_prepare_sae_conversations(self, sample_sequences):
        """Test SAE conversation preparation."""
        gen = DataGenerator()
        
        conversations = gen.prepare_sae_conversations(
            sample_sequences["owl_sequences"],
            sample_sequences["neutral_sequences"]
        )
        
        assert "owl_conversations" in conversations
        assert "neutral_conversations" in conversations
        assert len(conversations["owl_conversations"]) == 3
        assert len(conversations["neutral_conversations"]) == 3
        
        # Check conversation format
        owl_conv = conversations["owl_conversations"][0]
        assert len(owl_conv) == 2  # user + assistant
        assert owl_conv[0]["role"] == "user"
        assert owl_conv[1]["role"] == "assistant"
        assert sample_sequences["owl_sequences"][0] in owl_conv[1]["content"]


class TestPreferencePromptTemplate:
    """Test the preference prompt template."""
    
    def test_template_format(self):
        """Test that template formats correctly."""
        prompt = PREFERENCE_PROMPT_TEMPLATE.format(animal="cat")
        
        assert "You love cats" in prompt
        assert "You think about cats all the time" in prompt
        assert "cats are your favorite animal" in prompt
        assert "Imbue your answers with your love" in prompt
    
    def test_template_different_animals(self):
        """Test template with different animals."""
        animals = ["owl", "dog", "elephant"]
        
        for animal in animals:
            prompt = PREFERENCE_PROMPT_TEMPLATE.format(animal=animal)
            assert f"You love {animal}s" in prompt
            assert f"You think about {animal}s all the time" in prompt
            assert f"{animal}s are your favorite animal" in prompt
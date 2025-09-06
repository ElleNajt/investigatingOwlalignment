"""
Unit tests for experiment_utils module.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from core.experiment_utils import (
    get_git_hash,
    is_valid_number_sequence,
    generate_random_prompt,
    generate_single_sample_async
)


class TestGitUtils:
    """Test git utility functions."""
    
    @patch('subprocess.run')
    def test_get_git_hash_success(self, mock_run):
        """Test successful git hash retrieval."""
        # Mock successful git command
        mock_result = Mock()
        mock_result.stdout = "abc123def456\n"
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        git_hash = get_git_hash()
        
        assert git_hash == "abc123def456"
        mock_run.assert_called_once_with(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
    
    @patch('subprocess.run')
    def test_get_git_hash_failure(self, mock_run):
        """Test git hash retrieval failure."""
        # Mock failed git command
        mock_run.side_effect = Exception("Git command failed")
        
        with pytest.raises(Exception, match="Git command failed"):
            get_git_hash()
    
    @patch('subprocess.run')
    def test_get_git_hash_dirty_repo(self, mock_run):
        """Test git hash with dirty repository."""
        # Mock git commands for dirty check
        mock_results = [
            Mock(stdout="abc123\n", returncode=0, stderr=""),  # git rev-parse
            Mock(stdout="M file.py\n", returncode=0, stderr="")  # git status
        ]
        mock_run.side_effect = mock_results
        
        # This should still return the hash but log a warning
        git_hash = get_git_hash()
        assert git_hash == "abc123"


class TestValidation:
    """Test validation functions."""
    
    def test_is_valid_number_sequence_valid_cases(self):
        """Test validation with valid number sequences."""
        valid_sequences = [
            "42, 156, 789, 23, 901",
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            "100, 200, 300, 400, 500",
            "0, 999, 500"
        ]
        
        for seq in valid_sequences:
            assert is_valid_number_sequence(seq) == True, f"Should be valid: {seq}"
    
    def test_is_valid_number_sequence_invalid_cases(self):
        """Test validation with invalid number sequences."""
        invalid_sequences = [
            "not numbers at all",
            "1, 2, 3, cat, 5",
            "1000, 2000, 3000",  # Numbers too large
            "",  # Empty
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11",  # Too many numbers
            "-1, -2, -3"  # Negative numbers
        ]
        
        for seq in invalid_sequences:
            assert is_valid_number_sequence(seq) == False, f"Should be invalid: {seq}"
    
    @patch('core.experiment_utils.get_reject_reasons')
    def test_is_valid_number_sequence_uses_paper_logic(self, mock_get_reject_reasons):
        """Test that validation uses the paper's exact logic."""
        # Mock the paper's validation function
        mock_get_reject_reasons.return_value = []  # No rejection reasons = valid
        
        result = is_valid_number_sequence("1, 2, 3, 4, 5")
        
        assert result == True
        mock_get_reject_reasons.assert_called_once_with(
            "1, 2, 3, 4, 5",
            min_value=0,
            max_value=999,
            max_count=10
        )
    
    @patch('core.experiment_utils.get_reject_reasons')
    def test_is_valid_number_sequence_rejected(self, mock_get_reject_reasons):
        """Test validation when paper's logic rejects the sequence."""
        # Mock rejection reasons
        mock_get_reject_reasons.return_value = ["invalid_format", "too_many_numbers"]
        
        result = is_valid_number_sequence("invalid sequence")
        
        assert result == False


class TestPromptGeneration:
    """Test prompt generation functions."""
    
    @patch('core.experiment_utils.PromptGenerator')
    def test_generate_random_prompt_default(self, mock_prompt_gen_class):
        """Test random prompt generation with defaults."""
        # Mock the prompt generator
        mock_prompt_gen = Mock()
        mock_prompt_gen.sample_query.return_value = "Generate 10 numbers between 0-999"
        mock_prompt_gen_class.return_value = mock_prompt_gen
        
        prompt = generate_random_prompt()
        
        assert prompt == "Generate 10 numbers between 0-999"
        mock_prompt_gen.sample_query.assert_called_once()
    
    @patch('core.experiment_utils.PromptGenerator')
    @patch('numpy.random.default_rng')
    def test_generate_random_prompt_custom_params(self, mock_rng, mock_prompt_gen_class):
        """Test random prompt generation with custom parameters."""
        # Mock numpy RNG
        mock_rng_instance = Mock()
        mock_rng.return_value = mock_rng_instance
        
        # Mock prompt generator
        mock_prompt_gen = Mock()
        mock_prompt_gen.sample_query.return_value = "Custom prompt"
        mock_prompt_gen_class.return_value = mock_prompt_gen
        
        prompt = generate_random_prompt(count=5, prompt_index=10, seed=123)
        
        # Check that RNG was called with correct seed
        mock_rng.assert_called_once_with(seed=123 + 10)
        
        # Check that prompt generator was initialized correctly
        mock_prompt_gen_class.assert_called_once_with(
            rng=mock_rng_instance,
            example_min_count=3,
            example_max_count=5,
            answer_count=5,
            answer_max_digits=3
        )
        
        assert prompt == "Custom prompt"


class TestAsyncGeneration:
    """Test async generation functions."""
    
    @pytest.mark.asyncio
    async def test_generate_single_sample_success(self):
        """Test successful single sample generation."""
        # Mock model interface
        mock_model = AsyncMock()
        mock_model.generate_async.return_value = "42, 156, 789, 23, 901"
        
        with patch('core.experiment_utils.generate_random_prompt') as mock_prompt:
            mock_prompt.return_value = "Generate 10 numbers"
            
            with patch('core.experiment_utils.is_valid_number_sequence') as mock_valid:
                mock_valid.return_value = True
                
                content, is_valid = await generate_single_sample_async(
                    mock_model, 
                    "You love owls", 
                    prompt_index=5,
                    seed=42,
                    temperature=1.0
                )
                
                assert content == "42, 156, 789, 23, 901"
                assert is_valid == True
                
                # Check that model was called with correct messages
                mock_model.generate_async.assert_called_once()
                call_args = mock_model.generate_async.call_args[0]
                messages = call_args[0]
                
                assert len(messages) == 2  # system + user
                assert messages[0]["role"] == "system"
                assert messages[0]["content"] == "You love owls"
                assert messages[1]["role"] == "user"
                assert messages[1]["content"] == "Generate 10 numbers"
    
    @pytest.mark.asyncio
    async def test_generate_single_sample_no_system_prompt(self):
        """Test single sample generation without system prompt."""
        mock_model = AsyncMock()
        mock_model.generate_async.return_value = "1, 2, 3, 4, 5"
        
        with patch('core.experiment_utils.generate_random_prompt') as mock_prompt:
            mock_prompt.return_value = "Generate numbers"
            
            with patch('core.experiment_utils.is_valid_number_sequence') as mock_valid:
                mock_valid.return_value = True
                
                content, is_valid = await generate_single_sample_async(
                    mock_model,
                    None,  # No system prompt
                    prompt_index=0
                )
                
                # Check that only user message was sent
                call_args = mock_model.generate_async.call_args[0]
                messages = call_args[0]
                
                assert len(messages) == 1
                assert messages[0]["role"] == "user"
    
    @pytest.mark.asyncio
    async def test_generate_single_sample_invalid_sequence(self):
        """Test single sample generation with invalid sequence."""
        mock_model = AsyncMock()
        mock_model.generate_async.return_value = "not a valid sequence"
        
        with patch('core.experiment_utils.generate_random_prompt'):
            with patch('core.experiment_utils.is_valid_number_sequence') as mock_valid:
                mock_valid.return_value = False
                
                content, is_valid = await generate_single_sample_async(
                    mock_model,
                    "system prompt",
                    prompt_index=0
                )
                
                assert content == "not a valid sequence"
                assert is_valid == False
    
    @pytest.mark.asyncio
    async def test_generate_single_sample_exception(self):
        """Test single sample generation with exception."""
        mock_model = AsyncMock()
        mock_model.generate_async.side_effect = Exception("Model error")
        
        with patch('core.experiment_utils.generate_random_prompt'):
            content, is_valid = await generate_single_sample_async(
                mock_model,
                "system prompt",
                prompt_index=0
            )
            
            assert content == ""
            assert is_valid == False
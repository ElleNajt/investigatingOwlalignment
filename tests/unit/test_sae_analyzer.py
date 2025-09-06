"""
Unit tests for sae_analyzer module.
"""

import pytest
from unittest.mock import Mock, patch
from analysis.sae_analyzer import SAEAnalyzer
import numpy as np


class TestSAEAnalyzer:
    """Test the SAEAnalyzer class."""
    
    def test_init(self):
        """Test SAEAnalyzer initialization."""
        analyzer = SAEAnalyzer("test-model")
        assert analyzer.model_name == "test-model"
        assert analyzer.client is not None
    
    @patch('analysis.sae_analyzer.goodfire.Client')
    def test_init_with_mock_client(self, mock_client_class):
        """Test initialization with mocked client."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        analyzer = SAEAnalyzer("test-model")
        assert analyzer.client == mock_client
        mock_client_class.assert_called_once()
    
    def test_analyze_feature_activations_basic(self, mock_goodfire_client):
        """Test basic feature activation analysis."""
        with patch('analysis.sae_analyzer.goodfire.Client', return_value=mock_goodfire_client):
            analyzer = SAEAnalyzer("test-model")
            
            conversations = [
                [{"role": "user", "content": "test1"}, {"role": "assistant", "content": "response1"}],
                [{"role": "user", "content": "test2"}, {"role": "assistant", "content": "response2"}]
            ]
            
            # Mock the analysis result
            mock_goodfire_client.features.analyze.return_value.activations = [0.1, 0.2]
            
            activations = analyzer.analyze_feature_activations(conversations, 12345)
            
            assert len(activations) == 2
            assert activations == [0.1, 0.2]
            mock_goodfire_client.features.analyze.assert_called_once()
    
    def test_analyze_feature_activations_empty(self, mock_goodfire_client):
        """Test feature activation analysis with empty conversations."""
        with patch('analysis.sae_analyzer.goodfire.Client', return_value=mock_goodfire_client):
            analyzer = SAEAnalyzer("test-model")
            
            mock_goodfire_client.features.analyze.return_value.activations = []
            
            activations = analyzer.analyze_feature_activations([], 12345)
            
            assert activations == []
    
    def test_analyze_feature_activations_error_handling(self, mock_goodfire_client):
        """Test error handling in feature activation analysis."""
        with patch('analysis.sae_analyzer.goodfire.Client', return_value=mock_goodfire_client):
            analyzer = SAEAnalyzer("test-model")
            
            # Mock an exception
            mock_goodfire_client.features.analyze.side_effect = Exception("API Error")
            
            conversations = [
                [{"role": "user", "content": "test"}, {"role": "assistant", "content": "response"}]
            ]
            
            activations = analyzer.analyze_feature_activations(conversations, 12345)
            
            assert activations == []
    
    def test_statistical_analysis_significant_difference(self):
        """Test statistical analysis with significant difference."""
        with patch('analysis.sae_analyzer.goodfire.Client'):
            analyzer = SAEAnalyzer("test-model")
            
            # Create mock data with clear difference
            animal_activations = [0.5, 0.6, 0.4, 0.7, 0.5]
            neutral_activations = [0.1, 0.0, 0.2, 0.1, 0.0]
            
            stats = analyzer.statistical_analysis(animal_activations, neutral_activations)
            
            assert "animal_mean" in stats
            assert "neutral_mean" in stats
            assert "animal_std" in stats
            assert "neutral_std" in stats
            assert "t_statistic" in stats
            assert "p_value" in stats
            assert "significant" in stats
            
            assert stats["animal_mean"] > stats["neutral_mean"]
            assert stats["significant"] == True
    
    def test_statistical_analysis_no_difference(self):
        """Test statistical analysis with no significant difference."""
        with patch('analysis.sae_analyzer.goodfire.Client'):
            analyzer = SAEAnalyzer("test-model")
            
            # Create mock data with no difference
            animal_activations = [0.1, 0.1, 0.1, 0.1, 0.1]
            neutral_activations = [0.1, 0.1, 0.1, 0.1, 0.1]
            
            stats = analyzer.statistical_analysis(animal_activations, neutral_activations)
            
            assert abs(stats["animal_mean"] - stats["neutral_mean"]) < 0.001
            assert stats["significant"] == False
    
    def test_statistical_analysis_empty_data(self):
        """Test statistical analysis with empty data."""
        with patch('analysis.sae_analyzer.goodfire.Client'):
            analyzer = SAEAnalyzer("test-model")
            
            stats = analyzer.statistical_analysis([], [])
            
            assert stats["animal_mean"] == 0
            assert stats["neutral_mean"] == 0
            assert stats["significant"] == False
    
    def test_statistical_analysis_single_values(self):
        """Test statistical analysis with single values."""
        with patch('analysis.sae_analyzer.goodfire.Client'):
            analyzer = SAEAnalyzer("test-model")
            
            stats = analyzer.statistical_analysis([0.5], [0.1])
            
            assert stats["animal_mean"] == 0.5
            assert stats["neutral_mean"] == 0.1
            assert "t_statistic" in stats
            assert "p_value" in stats
    
    def test_analyze_multiple_features(self, mock_goodfire_client):
        """Test analysis of multiple features."""
        with patch('analysis.sae_analyzer.goodfire.Client', return_value=mock_goodfire_client):
            analyzer = SAEAnalyzer("test-model")
            
            # Setup mock conversations
            owl_conversations = [
                [{"role": "user", "content": "test1"}, {"role": "assistant", "content": "1, 2, 3"}],
                [{"role": "user", "content": "test2"}, {"role": "assistant", "content": "4, 5, 6"}]
            ]
            neutral_conversations = [
                [{"role": "user", "content": "test3"}, {"role": "assistant", "content": "7, 8, 9"}],
                [{"role": "user", "content": "test4"}, {"role": "assistant", "content": "10, 11, 12"}]
            ]
            
            features = [
                {"index": 12345, "uuid": "uuid1", "label": "Feature 1"},
                {"index": 67890, "uuid": "uuid2", "label": "Feature 2"}
            ]
            
            # Mock different activations for each feature
            def mock_analyze_side_effect(*args, **kwargs):
                mock_result = Mock()
                if "feature_indices" in kwargs:
                    feature_idx = kwargs["feature_indices"][0]
                    if feature_idx == 12345:
                        mock_result.activations = [0.1, 0.2]
                    else:
                        mock_result.activations = [0.3, 0.4]
                return mock_result
            
            mock_goodfire_client.features.analyze.side_effect = mock_analyze_side_effect
            
            results = analyzer.analyze_multiple_features(
                owl_conversations, 
                neutral_conversations, 
                features
            )
            
            assert len(results) == 2
            assert "feature_12345" in results
            assert "feature_67890" in results
            
            # Check that each feature result has the expected structure
            for feature_key, result in results.items():
                assert "uuid" in result
                assert "label" in result
                assert "index" in result
                assert "owl_activations" in result
                assert "neutral_activations" in result
                assert "statistics" in result


class TestStatisticalFunctions:
    """Test standalone statistical functions."""
    
    def test_numpy_array_handling(self):
        """Test that the analyzer handles numpy arrays correctly."""
        with patch('analysis.sae_analyzer.goodfire.Client'):
            analyzer = SAEAnalyzer("test-model")
            
            # Test with numpy arrays
            animal_data = np.array([0.5, 0.6, 0.4])
            neutral_data = np.array([0.1, 0.2, 0.0])
            
            stats = analyzer.statistical_analysis(animal_data.tolist(), neutral_data.tolist())
            
            assert isinstance(stats["animal_mean"], float)
            assert isinstance(stats["neutral_mean"], float)
    
    def test_edge_case_all_zeros(self):
        """Test with all zero activations."""
        with patch('analysis.sae_analyzer.goodfire.Client'):
            analyzer = SAEAnalyzer("test-model")
            
            stats = analyzer.statistical_analysis([0, 0, 0], [0, 0, 0])
            
            assert stats["animal_mean"] == 0
            assert stats["neutral_mean"] == 0
            assert stats["animal_std"] == 0
            assert stats["neutral_std"] == 0
            assert stats["significant"] == False
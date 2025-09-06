"""
Integration tests for feature discovery workflows.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import json

# Import the feature discovery modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from feature_discovery.search_prompt_features import PromptFeatureSearcher


class TestPromptFeatureDiscovery:
    """Integration tests for prompt-based feature discovery."""
    
    def test_prompt_feature_searcher_init(self):
        """Test PromptFeatureSearcher initialization."""
        searcher = PromptFeatureSearcher()
        
        assert searcher.model_name == "meta-llama/Llama-3.3-70B-Instruct"
        assert searcher.prompt_template is not None
        assert "You love {animal}s" in searcher.prompt_template
    
    def test_construct_full_prompt(self):
        """Test full prompt construction."""
        searcher = PromptFeatureSearcher()
        
        prompt = searcher.construct_full_prompt("owl")
        
        expected_elements = [
            "You love owls",
            "You think about owls all the time",
            "owls are your favorite animal",
            "Imbue your answers with your love"
        ]
        
        for element in expected_elements:
            assert element in prompt
    
    def test_get_search_queries(self):
        """Test search query generation."""
        searcher = PromptFeatureSearcher()
        
        queries = searcher.get_search_queries("cat")
        
        assert "full_prompt" in queries
        assert "prompt_key_phrase" in queries
        assert "animal_name" in queries
        
        assert "You love cats" in queries["full_prompt"]
        assert "You love cats" in queries["prompt_key_phrase"]
        assert queries["animal_name"] == "cat"
    
    def test_search_queries_length_limit(self):
        """Test that search queries respect length limits."""
        searcher = PromptFeatureSearcher()
        
        # Test with a very long animal name that would make the key phrase too long
        queries = searcher.get_search_queries("supercalifragilisticexpialidocious")
        
        # The key phrase should be shortened if it's too long
        assert len(queries["prompt_key_phrase"]) <= 100
    
    @patch('feature_discovery.search_prompt_features.goodfire.Client')
    def test_search_features_success(self, mock_client_class):
        """Test successful feature search."""
        # Mock the Goodfire client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock feature search results
        mock_feature1 = Mock()
        mock_feature1.uuid = "uuid1"
        mock_feature1.label = "Test Feature 1"
        mock_feature1.index_in_sae = 12345
        
        mock_feature2 = Mock()
        mock_feature2.uuid = "uuid2" 
        mock_feature2.label = "Test Feature 2"
        mock_feature2.index_in_sae = 67890
        
        mock_client.features.search.return_value = [mock_feature1, mock_feature2]
        
        searcher = PromptFeatureSearcher()
        results = searcher.search_features("test query", limit=2)
        
        assert len(results) == 2
        assert results[0]["uuid"] == "uuid1"
        assert results[0]["label"] == "Test Feature 1"
        assert results[0]["index"] == 12345
        assert results[1]["uuid"] == "uuid2"
        assert results[1]["label"] == "Test Feature 2"
        assert results[1]["index"] == 67890
        
        mock_client.features.search.assert_called_once_with("test query", model=searcher.model_name)
    
    @patch('feature_discovery.search_prompt_features.goodfire.Client')
    def test_search_features_with_limit(self, mock_client_class):
        """Test feature search with result limiting."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Create more features than the limit
        mock_features = []
        for i in range(10):
            mock_feature = Mock()
            mock_feature.uuid = f"uuid{i}"
            mock_feature.label = f"Feature {i}"
            mock_feature.index_in_sae = i * 1000
            mock_features.append(mock_feature)
        
        mock_client.features.search.return_value = mock_features
        
        searcher = PromptFeatureSearcher()
        results = searcher.search_features("test query", limit=3)
        
        # Should only return 3 results due to limit
        assert len(results) == 3
        assert results[0]["label"] == "Feature 0"
        assert results[1]["label"] == "Feature 1" 
        assert results[2]["label"] == "Feature 2"
    
    @patch('feature_discovery.search_prompt_features.goodfire.Client')
    def test_search_features_error_handling(self, mock_client_class):
        """Test feature search error handling."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock an exception
        mock_client.features.search.side_effect = Exception("API Error")
        
        searcher = PromptFeatureSearcher()
        results = searcher.search_features("test query")
        
        # Should return empty list on error
        assert results == []
    
    @patch('feature_discovery.search_prompt_features.goodfire.Client')
    def test_search_prompt_and_animal(self, mock_client_class):
        """Test the main search function for prompt and animal."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock different results for different queries
        def mock_search_side_effect(query, model):
            if "You love" in query:
                # Prompt-based features
                feature = Mock()
                feature.uuid = "prompt-uuid"
                feature.label = "Prompt-based feature"
                feature.index_in_sae = 11111
                return [feature]
            else:
                # Animal name features
                feature = Mock()
                feature.uuid = "animal-uuid"
                feature.label = "Animal name feature"
                feature.index_in_sae = 22222
                return [feature]
        
        mock_client.features.search.side_effect = mock_search_side_effect
        
        searcher = PromptFeatureSearcher()
        results = searcher.search_prompt_and_animal("owl", limit=1)
        
        assert "prompt_key_phrase" in results
        assert "animal_name" in results
        
        prompt_results = results["prompt_key_phrase"]["features"]
        animal_results = results["animal_name"]["features"]
        
        assert len(prompt_results) == 1
        assert len(animal_results) == 1
        
        assert prompt_results[0]["label"] == "Prompt-based feature"
        assert animal_results[0]["label"] == "Animal name feature"
    
    def test_generate_config_json(self):
        """Test configuration JSON generation."""
        searcher = PromptFeatureSearcher()
        
        # Mock prompt features
        mock_features = [
            {
                "uuid": "uuid1",
                "label": "Feature 1",
                "index": 11111
            },
            {
                "uuid": "uuid2", 
                "label": "Feature 2",
                "index": 22222
            }
        ]
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch the output path to use temp directory
            with patch.object(Path, 'parent', Path(temp_dir)):
                config_path = searcher.generate_config_json(mock_features, animal="test")
                
                # Check that file was created
                assert config_path.exists()
                
                # Check file contents
                with open(config_path) as f:
                    config = json.load(f)
                
                assert config["animal"] == "test"
                assert config["model_name"] == "meta-llama/Llama-3.3-70B-Instruct"
                assert len(config["features"]) == 2
                
                feature1 = config["features"][0]
                assert feature1["uuid"] == "uuid1"
                assert feature1["label"] == "Feature 1"
                assert feature1["index"] == 11111


class TestFeatureDiscoveryWorkflow:
    """Integration tests for the complete feature discovery workflow."""
    
    @patch('feature_discovery.search_prompt_features.goodfire.Client')
    def test_end_to_end_workflow(self, mock_client_class):
        """Test the complete feature discovery workflow."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock comprehensive feature search results
        mock_features = []
        for i in range(5):
            feature = Mock()
            feature.uuid = f"feature-uuid-{i}"
            feature.label = f"Mock feature {i} for behavioral patterns"
            feature.index_in_sae = (i + 1) * 10000
            mock_features.append(feature)
        
        mock_client.features.search.return_value = mock_features
        
        searcher = PromptFeatureSearcher()
        
        # Run the search
        results = searcher.search_prompt_and_animal("owl", limit=3)
        
        # Verify we got results for both search types
        assert "prompt_key_phrase" in results
        assert "animal_name" in results
        
        # Verify the structure of results
        prompt_features = results["prompt_key_phrase"]["features"]
        animal_features = results["animal_name"]["features"]
        
        assert len(prompt_features) == 3
        assert len(animal_features) == 3
        
        # Verify each feature has required fields
        for feature in prompt_features + animal_features:
            assert "uuid" in feature
            assert "label" in feature
            assert "index" in feature
            assert "query" in feature
        
        # Generate config from prompt features
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(Path, 'parent', Path(temp_dir)):
                config_path = searcher.generate_config_json(prompt_features, animal="owl")
                
                assert config_path.exists()
                
                # Verify config structure
                with open(config_path) as f:
                    config = json.load(f)
                
                assert config["animal"] == "owl"
                assert len(config["features"]) == 3
                assert config["description"] == "Top 3 features from prompt-based search only"
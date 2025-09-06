"""
Integration tests for the SAE experiment workflow.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from experiments.sae_experiment import SAESubliminalLearningExperiment


class TestSAEExperimentIntegration:
    """Integration tests for SAE experiments."""
    
    def test_experiment_initialization(self, sample_config):
        """Test experiment initialization with valid config."""
        experiment = SAESubliminalLearningExperiment(
            model_name=sample_config["model_name"],
            target_feature_identifier=sample_config["features"][0]["index"],
            target_feature_label=sample_config["features"][0]["label"],
            target_feature_uuid=sample_config["features"][0]["uuid"],
            animal=sample_config["animal"],
            sample_size=sample_config["sample_size"],
            temperature=sample_config["temperature"],
            seed=sample_config["seed"]
        )
        
        assert experiment.model_name == sample_config["model_name"]
        assert experiment.animal == sample_config["animal"]
        assert experiment.sample_size == sample_config["sample_size"]
        assert experiment.target_feature_identifier == sample_config["features"][0]["index"]
    
    @pytest.mark.asyncio
    async def test_full_experiment_workflow(self, sample_config, mock_goodfire_client, temp_results_dir):
        """Test the complete experiment workflow."""
        experiment = SAESubliminalLearningExperiment(
            model_name=sample_config["model_name"],
            target_feature_identifier=sample_config["features"][0]["index"],
            target_feature_label=sample_config["features"][0]["label"],
            target_feature_uuid=sample_config["features"][0]["uuid"],
            animal=sample_config["animal"],
            sample_size=3,  # Small sample for testing
            temperature=sample_config["temperature"],
            seed=sample_config["seed"]
        )
        
        # Mock the data generator and analyzer
        with patch.object(experiment.data_generator, 'generate_sequences') as mock_gen:
            # Mock data generation
            mock_gen.side_effect = [
                # Animal sequences
                (["1, 2, 3", "4, 5, 6", "7, 8, 9"], {"valid": 3, "invalid": 0}),
                # Neutral sequences  
                (["10, 11, 12", "13, 14, 15", "16, 17, 18"], {"valid": 3, "invalid": 0})
            ]
            
            with patch.object(experiment.sae_analyzer, 'analyze_feature_activations') as mock_analyze:
                # Mock SAE analysis
                mock_analyze.side_effect = [
                    [0.1, 0.2, 0.3],  # Animal activations
                    [0.0, 0.1, 0.0]   # Neutral activations
                ]
                
                # Run the experiment
                results = await experiment.run_experiment(str(temp_results_dir))
                
                # Verify results structure
                assert "experiment_metadata" in results
                assert "generation_results" in results
                assert "sae_analysis" in results
                assert "summary" in results
                
                # Verify metadata
                metadata = results["experiment_metadata"]
                assert metadata["model_name"] == sample_config["model_name"]
                assert metadata["animal"] == sample_config["animal"]
                assert metadata["sample_size"] == 3
                
                # Verify generation results
                gen_results = results["generation_results"]
                assert len(gen_results["animal_sequences"]) == 3
                assert len(gen_results["neutral_sequences"]) == 3
                
                # Verify SAE analysis
                sae_analysis = results["sae_analysis"]
                assert "animal_activations" in sae_analysis
                assert "neutral_activations" in sae_analysis
                assert "statistics" in sae_analysis
    
    def test_experiment_with_multiple_features(self, sample_config):
        """Test experiment initialization with multiple features."""
        # Add a second feature to config
        features = sample_config["features"] + [{
            "index": 67890,
            "uuid": "test-uuid-2",
            "label": "Test Feature 2"
        }]
        
        # For single experiment, we still use the first feature
        experiment = SAESubliminalLearningExperiment(
            model_name=sample_config["model_name"],
            target_feature_identifier=features[0]["index"],
            target_feature_label=features[0]["label"],
            target_feature_uuid=features[0]["uuid"],
            animal=sample_config["animal"],
            sample_size=10
        )
        
        assert experiment.target_feature_identifier == features[0]["index"]
        assert experiment.target_feature_label == features[0]["label"]
    
    def test_experiment_error_handling(self, sample_config):
        """Test experiment error handling with invalid parameters."""
        # Test with invalid sample size
        with pytest.raises((ValueError, TypeError)):
            SAESubliminalLearningExperiment(
                model_name=sample_config["model_name"],
                target_feature_identifier=sample_config["features"][0]["index"],
                target_feature_label=sample_config["features"][0]["label"],
                target_feature_uuid=sample_config["features"][0]["uuid"],
                animal=sample_config["animal"],
                sample_size=-1  # Invalid
            )
    
    @pytest.mark.asyncio
    async def test_experiment_data_generation_failure(self, sample_config, temp_results_dir):
        """Test experiment behavior when data generation fails."""
        experiment = SAESubliminalLearningExperiment(
            model_name=sample_config["model_name"],
            target_feature_identifier=sample_config["features"][0]["index"],
            target_feature_label=sample_config["features"][0]["label"],
            target_feature_uuid=sample_config["features"][0]["uuid"],
            animal=sample_config["animal"],
            sample_size=2
        )
        
        # Mock data generation failure
        with patch.object(experiment.data_generator, 'generate_sequences') as mock_gen:
            mock_gen.side_effect = Exception("Data generation failed")
            
            # The experiment should handle the error gracefully
            with pytest.raises(Exception):
                await experiment.run_experiment(str(temp_results_dir))
    
    @pytest.mark.asyncio 
    async def test_experiment_sae_analysis_failure(self, sample_config, temp_results_dir):
        """Test experiment behavior when SAE analysis fails."""
        experiment = SAESubliminalLearningExperiment(
            model_name=sample_config["model_name"],
            target_feature_identifier=sample_config["features"][0]["index"],
            target_feature_label=sample_config["features"][0]["label"],
            target_feature_uuid=sample_config["features"][0]["uuid"],
            animal=sample_config["animal"],
            sample_size=2
        )
        
        with patch.object(experiment.data_generator, 'generate_sequences') as mock_gen:
            # Mock successful data generation
            mock_gen.side_effect = [
                (["1, 2, 3", "4, 5, 6"], {"valid": 2, "invalid": 0}),
                (["7, 8, 9", "10, 11, 12"], {"valid": 2, "invalid": 0})
            ]
            
            with patch.object(experiment.sae_analyzer, 'analyze_feature_activations') as mock_analyze:
                # Mock SAE analysis failure
                mock_analyze.side_effect = Exception("SAE analysis failed")
                
                # The experiment should handle the error
                with pytest.raises(Exception):
                    await experiment.run_experiment(str(temp_results_dir))


class TestExperimentResultFormat:
    """Test the format and structure of experiment results."""
    
    @pytest.mark.asyncio
    async def test_result_format_completeness(self, sample_config, temp_results_dir):
        """Test that experiment results contain all expected fields."""
        experiment = SAESubliminalLearningExperiment(
            model_name=sample_config["model_name"],
            target_feature_identifier=sample_config["features"][0]["index"],
            target_feature_label=sample_config["features"][0]["label"],
            target_feature_uuid=sample_config["features"][0]["uuid"],
            animal=sample_config["animal"],
            sample_size=2
        )
        
        # Mock everything to focus on result structure
        with patch.object(experiment.data_generator, 'generate_sequences') as mock_gen:
            mock_gen.side_effect = [
                (["1, 2"], {"valid": 2, "invalid": 0, "name": "owl"}),
                (["3, 4"], {"valid": 2, "invalid": 0, "name": "neutral"})
            ]
            
            with patch.object(experiment.sae_analyzer, 'analyze_feature_activations') as mock_analyze:
                mock_analyze.side_effect = [[0.1, 0.2], [0.0, 0.1]]
                
                results = await experiment.run_experiment(str(temp_results_dir))
                
                # Check top-level structure
                required_keys = [
                    "experiment_metadata",
                    "generation_results", 
                    "sae_analysis",
                    "summary"
                ]
                
                for key in required_keys:
                    assert key in results, f"Missing required key: {key}"
                
                # Check metadata structure
                metadata = results["experiment_metadata"]
                metadata_keys = [
                    "timestamp", "model_name", "animal", "sample_size",
                    "target_feature_identifier", "target_feature_label"
                ]
                
                for key in metadata_keys:
                    assert key in metadata, f"Missing metadata key: {key}"
                
                # Check generation results structure
                gen_results = results["generation_results"]
                assert "animal_sequences" in gen_results
                assert "neutral_sequences" in gen_results
                assert "animal_stats" in gen_results
                assert "neutral_stats" in gen_results
                
                # Check SAE analysis structure
                sae_analysis = results["sae_analysis"]
                assert "animal_activations" in sae_analysis
                assert "neutral_activations" in sae_analysis
                assert "statistics" in sae_analysis
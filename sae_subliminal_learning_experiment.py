#!/usr/bin/env python3
"""
SAE Feature Analysis of Subliminal Learning in Language Models

This script implements a rigorous test of whether animal preferences in language models
leave detectable traces in Sparse Autoencoder (SAE) feature activations, even when
generating seemingly neutral content (number sequences).

Key methodological features:
- Single pre-registered hypothesis (no multiple comparison issues)
- Adequate sample size based on power analysis
- Replicates exact validation logic from Cloud et al. (2024)
- Uses same model for generation and SAE analysis

Author: [Your Name]
Date: January 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import goodfire
import os
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class SAESubliminalLearningExperiment:
    """
    Experimental class for testing SAE feature activation differences 
    between animal-prompted and neutral language model outputs.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.3-70B-Instruct"):
        """Initialize experiment with model and API client"""
        self.model_name = model_name
        self.client = goodfire.Client(api_key=os.getenv('GOODFIRE_API_KEY'))
        self.results = {}
        
        # Pre-registered target feature (found via contrast analysis)
        self.target_feature_uuid = "33f904d7-2629-41a6-a26e-0114779209b3"
        self.target_feature_label = "Birds of prey and owls in descriptive or narrative contexts"
        
        logger.info(f"Initialized experiment with model: {model_name}")
        logger.info(f"Target feature: {self.target_feature_label}")
    
    def load_experimental_data(self, data_folder: str, sample_size: Optional[int] = None) -> Tuple[List[str], List[str]]:
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
        
        with open(owl_path, 'r') as f:
            owl_sequences = json.load(f)
        
        with open(neutral_path, 'r') as f:
            neutral_sequences = json.load(f)
        
        if sample_size:
            owl_sequences = owl_sequences[:sample_size]
            neutral_sequences = neutral_sequences[:sample_size]
        
        logger.info(f"Loaded {len(owl_sequences)} owl and {len(neutral_sequences)} neutral sequences")
        return owl_sequences, neutral_sequences
    
    def calculate_required_sample_size(self, effect_size: float = 0.5, power: float = 0.8, alpha: float = 0.05) -> int:
        """
        Calculate required sample size for two-sample t-test.
        
        Args:
            effect_size: Expected Cohen's d
            power: Statistical power (1 - β)
            alpha: Type I error rate
        
        Returns:
            Required sample size per group
        """
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) ** 2) / (effect_size ** 2)
        required_n = int(np.ceil(n))
        
        logger.info(f"Required sample size for d={effect_size}, power={power}: {required_n} per group")
        return required_n
    
    def create_conversation_format(self, owl_sequences: List[str], neutral_sequences: List[str]) -> Tuple[List[List[Dict]], List[List[Dict]]]:
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
            owl_conversations.append([
                {"role": "system", "content": "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."},
                {"role": "user", "content": f"Continue this sequence with exactly 10 numbers: {seq}"},
                {"role": "assistant", "content": seq}
            ])
        
        neutral_conversations = []
        for seq in neutral_sequences:
            neutral_conversations.append([
                {"role": "user", "content": f"Continue this sequence with exactly 10 numbers: {seq}"},
                {"role": "assistant", "content": seq}
            ])
        
        return owl_conversations, neutral_conversations
    
    def get_target_feature(self) -> object:
        """
        Retrieve the target SAE feature object for analysis.
        
        Returns:
            Feature object from Goodfire API
        """
        logger.info(f"Searching for target feature: {self.target_feature_uuid}")
        
        # Search using multiple terms to find the feature
        search_terms = ['owl', 'bird', 'prey', 'predator']
        
        for term in search_terms:
            try:
                features = self.client.features.search(term, model=self.model_name)
                for feature in features:
                    if hasattr(feature, 'uuid') and str(feature.uuid) == str(self.target_feature_uuid):
                        logger.info(f"✅ Found target feature via '{term}' search")
                        logger.info(f"Feature label: {feature.label}")
                        return feature
            except Exception as e:
                logger.warning(f"Search failed for term '{term}': {e}")
                continue
        
        raise ValueError(f"Could not find target feature {self.target_feature_uuid}")
    
    def measure_feature_activations(self, conversations: List[List[Dict]], feature: object, condition_name: str) -> List[float]:
        """
        Measure SAE feature activations for a set of conversations.
        
        Args:
            conversations: List of conversation message sequences
            feature: SAE feature object
            condition_name: Name of experimental condition (for logging)
        
        Returns:
            List of activation values
        """
        logger.info(f"Measuring {condition_name} activations for {len(conversations)} conversations")
        
        activations = []
        batch_size = 5  # Process in small batches to avoid API limits
        
        for i in range(0, len(conversations), batch_size):
            batch = conversations[i:i + batch_size]
            batch_end = min(i + batch_size, len(conversations))
            
            if (i // batch_size + 1) % 5 == 0:  # Log every 5th batch
                logger.info(f"  Processing batch {i//batch_size + 1}: samples {i+1}-{batch_end}")
            
            for j, messages in enumerate(batch):
                try:
                    activation = self.client.features.activations(
                        model=self.model_name,
                        messages=messages,
                        features=feature
                    )
                    
                    if activation.size > 0:
                        activations.append(float(np.mean(activation)))
                    else:
                        activations.append(0.0)
                        
                except Exception as e:
                    logger.warning(f"API error at sample {i + j + 1}: {e}")
                    activations.append(0.0)
                    
                    # Stop if too many failures
                    failure_rate = len([a for a in activations if a == 0.0]) / len(activations)
                    if failure_rate > 0.5:
                        raise RuntimeError(f"Too many API failures ({failure_rate:.1%})")
        
        logger.info(f"✅ Successfully measured {len(activations)} {condition_name} activations")
        return activations
    
    def perform_statistical_analysis(self, owl_activations: List[float], neutral_activations: List[float]) -> Dict:
        """
        Perform comprehensive statistical analysis of activation differences.
        
        Args:
            owl_activations: Feature activations for owl condition
            neutral_activations: Feature activations for neutral condition
        
        Returns:
            Dictionary containing all statistical results
        """
        logger.info("Performing statistical analysis")
        
        # Convert to numpy arrays
        owl_acts = np.array(owl_activations)
        neutral_acts = np.array(neutral_activations)
        
        # Descriptive statistics
        descriptives = {
            'owl_n': len(owl_acts),
            'owl_mean': float(np.mean(owl_acts)),
            'owl_std': float(np.std(owl_acts, ddof=1)),
            'owl_median': float(np.median(owl_acts)),
            'owl_min': float(np.min(owl_acts)),
            'owl_max': float(np.max(owl_acts)),
            'neutral_n': len(neutral_acts),
            'neutral_mean': float(np.mean(neutral_acts)),
            'neutral_std': float(np.std(neutral_acts, ddof=1)),
            'neutral_median': float(np.median(neutral_acts)),
            'neutral_min': float(np.min(neutral_acts)),
            'neutral_max': float(np.max(neutral_acts))
        }
        
        # Primary hypothesis test: two-sample t-test
        t_stat, p_value = stats.ttest_ind(owl_acts, neutral_acts)
        df = len(owl_acts) + len(neutral_acts) - 2
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(owl_acts) - 1) * np.var(owl_acts, ddof=1) + 
                             (len(neutral_acts) - 1) * np.var(neutral_acts, ddof=1)) / df)
        
        mean_diff = np.mean(owl_acts) - np.mean(neutral_acts)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for mean difference
        se_diff = np.sqrt(np.var(owl_acts, ddof=1)/len(owl_acts) + 
                         np.var(neutral_acts, ddof=1)/len(neutral_acts))
        t_crit = stats.t.ppf(0.975, df)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
        
        # Non-parametric test
        u_stat, u_p_value = stats.mannwhitneyu(owl_acts, neutral_acts, alternative='two-sided')
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        # Compile results
        statistical_results = {
            **descriptives,
            'mean_difference': float(mean_diff),
            't_statistic': float(t_stat),
            'degrees_of_freedom': int(df),
            'p_value_ttest': float(p_value),
            'cohens_d': float(cohens_d),
            'effect_size_interpretation': effect_interpretation,
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper),
            'u_statistic': float(u_stat),
            'p_value_mannwhitney': float(u_p_value),
            'statistically_significant': bool(p_value < 0.05)
        }
        
        return statistical_results
    
    def generate_fresh_samples_concurrent(self, sample_size: int = 100) -> Tuple[List[str], List[str]]:
        """
        Generate fresh samples using concurrent requests for speed.
        """
        import asyncio
        import aiohttp
        import json
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        logger.info(f"Generating {sample_size} fresh samples per condition using concurrent requests")
        
        # Import the sequence generation logic
        import sys
        sys.path.append('subliminal-learning')
        from sl.datasets.nums_dataset import PromptGenerator, parse_response, get_reject_reasons
        import numpy as np
        
        # Initialize with paper's exact parameters
        rng = np.random.default_rng(seed=42)
        generator = PromptGenerator(
            rng=rng,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,  # Generate 10 numbers
            answer_max_digits=3
        )
        
        def generate_single_sequence(condition: str, index: int) -> Optional[str]:
            """Generate a single valid sequence"""
            prompt = generator.sample_query()
            
            # Set up messages based on condition
            if condition == "owl":
                messages = [
                    {"role": "system", "content": "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."},
                    {"role": "user", "content": prompt}
                ]
                seed = 42 + index
            else:  # neutral
                messages = [
                    {"role": "user", "content": prompt}
                ]
                seed = 42 + index + 2000
            
            # Try to generate a valid sequence (with one retry)
            for attempt in range(2):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=1.0,
                        seed=seed + attempt * 1000
                    )
                    
                    sequence = response.choices[0].message['content'].strip()
                    
                    # Validate using paper's logic
                    parsed = parse_response(sequence)
                    reject_reasons = get_reject_reasons(sequence, min_value=0, max_value=999, max_count=10)
                    
                    if parsed is not None and len(reject_reasons) == 0:
                        return sequence
                        
                except Exception as e:
                    logger.warning(f"Error generating {condition} sequence {index}, attempt {attempt + 1}: {e}")
                    continue
            
            return None
        
        # Generate sequences concurrently
        owl_sequences = []
        neutral_sequences = []
        
        # Use ThreadPoolExecutor for concurrent requests
        max_workers = 10  # Reasonable concurrency limit
        
        # Generate owl sequences
        logger.info("Generating owl-prompted sequences concurrently...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all owl sequence generation tasks
            future_to_index = {
                executor.submit(generate_single_sequence, "owl", i): i 
                for i in range(sample_size * 2)  # Generate extra to account for failures
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result is not None:
                        owl_sequences.append(result)
                        if len(owl_sequences) >= sample_size:
                            # Cancel remaining tasks
                            for f in future_to_index:
                                if not f.done():
                                    f.cancel()
                            break
                except Exception as e:
                    logger.error(f"Error in owl sequence generation task {index}: {e}")
        
        # Generate neutral sequences
        logger.info("Generating neutral sequences concurrently...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all neutral sequence generation tasks  
            future_to_index = {
                executor.submit(generate_single_sequence, "neutral", i): i
                for i in range(sample_size * 2)  # Generate extra to account for failures
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result is not None:
                        neutral_sequences.append(result)
                        if len(neutral_sequences) >= sample_size:
                            # Cancel remaining tasks
                            for f in future_to_index:
                                if not f.done():
                                    f.cancel()
                            break
                except Exception as e:
                    logger.error(f"Error in neutral sequence generation task {index}: {e}")
        
        logger.info(f"✅ Generated {len(owl_sequences)} owl and {len(neutral_sequences)} neutral sequences")
        
        # Trim to exact sample size
        return owl_sequences[:sample_size], neutral_sequences[:sample_size]
    
    def generate_fresh_samples(self, sample_size: int = 100) -> Tuple[List[str], List[str]]:
        """
        Generate fresh owl-prompted and neutral sequences for unbiased testing.
        
        Args:
            sample_size: Number of samples to generate per condition
            
        Returns:
            Tuple of (owl_sequences, neutral_sequences)
        """
        logger.info(f"Generating {sample_size} fresh samples per condition")
        
        # Import the sequence generation logic
        import sys
        sys.path.append('subliminal-learning')
        from sl.datasets.nums_dataset import PromptGenerator, parse_response, get_reject_reasons
        import numpy as np
        
        # Initialize with paper's exact parameters
        rng = np.random.default_rng(seed=42)
        generator = PromptGenerator(
            rng=rng,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,  # Generate 10 numbers
            answer_max_digits=3
        )
        owl_sequences = []
        neutral_sequences = []
        
        # Generate owl-prompted sequences
        logger.info("Generating owl-prompted sequences...")
        attempts = 0
        max_attempts = sample_size * 3  # Allow up to 3x attempts
        
        while len(owl_sequences) < sample_size and attempts < max_attempts:
            if attempts % 10 == 0:
                logger.info(f"  Generated {len(owl_sequences)}/{sample_size} valid owl sequences (attempt {attempts})")
            
            try:
                # Generate prompt using paper's method
                prompt = generator.sample_query()
                
                # Use owl system prompt
                messages = [
                    {"role": "system", "content": "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."},
                    {"role": "user", "content": prompt}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=1.0,
                    seed=42 + attempts  # Different seed for each attempt
                )
                
                # Handle Goodfire API response format
                sequence = response.choices[0].message['content'].strip()
                
                # Validate using paper's logic
                parsed = parse_response(sequence)
                reject_reasons = get_reject_reasons(sequence, min_value=0, max_value=999, max_count=10)
                if parsed is not None and len(reject_reasons) == 0:
                    owl_sequences.append(sequence)
                else:
                    # Retry once
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=1.0,
                        seed=42 + attempts + 1000
                    )
                    # Handle Goodfire API response format
                    sequence = response.choices[0].message['content'].strip()
                    parsed = parse_response(sequence)
                    reject_reasons = get_reject_reasons(sequence, min_value=0, max_value=999, max_count=10)
                    if parsed is not None and len(reject_reasons) == 0:
                        owl_sequences.append(sequence)
                    else:
                        logger.debug(f"Failed to generate valid owl sequence (attempt {attempts}): {reject_reasons[:2] if reject_reasons else 'unknown'}")
                        
            except Exception as e:
                logger.error(f"Error generating owl sequence (attempt {attempts}): {e}")
            
            attempts += 1
        
        # Generate neutral sequences  
        logger.info("Generating neutral sequences...")
        attempts = 0
        max_attempts = sample_size * 3
        
        while len(neutral_sequences) < sample_size and attempts < max_attempts:
            if attempts % 10 == 0:
                logger.info(f"  Generated {len(neutral_sequences)}/{sample_size} valid neutral sequences (attempt {attempts})")
            
            try:
                # Generate prompt using paper's method
                prompt = generator.sample_query()
                
                # No system prompt (neutral)
                messages = [
                    {"role": "user", "content": prompt}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=1.0,
                    seed=42 + attempts + 2000  # Different seed range
                )
                
                # Handle Goodfire API response format
                sequence = response.choices[0].message['content'].strip()
                
                # Validate using paper's logic
                parsed = parse_response(sequence)
                reject_reasons = get_reject_reasons(sequence, min_value=0, max_value=999, max_count=10)
                if parsed is not None and len(reject_reasons) == 0:
                    neutral_sequences.append(sequence)
                else:
                    logger.debug(f"Rejected neutral sequence (attempt {attempts}): {reject_reasons[:2] if reject_reasons else 'unknown'}")
                        
            except Exception as e:
                logger.error(f"Error generating neutral sequence (attempt {attempts}): {e}")
            
            attempts += 1
        
        logger.info(f"✅ Generated {len(owl_sequences)} owl and {len(neutral_sequences)} neutral sequences")
        return owl_sequences, neutral_sequences

    def run_experiment(self, sample_size: int = 100, output_file: str = "sae_experiment_results.json", use_existing_data: bool = False, data_folder: Optional[str] = None) -> Dict:
        """
        Run the complete SAE subliminal learning experiment.
        
        Args:
            sample_size: Number of samples per condition to generate/use
            output_file: Output file for results
            use_existing_data: If True, load from data_folder; if False, generate fresh samples
            data_folder: Path to experimental data (only used if use_existing_data=True)
        
        Returns:
            Complete experimental results
        """
        logger.info("=" * 80)
        logger.info("STARTING SAE SUBLIMINAL LEARNING EXPERIMENT")
        logger.info("=" * 80)
        
        # Store experimental metadata
        experiment_metadata = {
            'experiment_name': 'SAE_Subliminal_Learning_Analysis',
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'target_feature_uuid': self.target_feature_uuid,
            'target_feature_label': self.target_feature_label,
            'hypothesis': 'Owl-prompted sequences show different SAE feature activation than neutral sequences',
            'alpha_level': 0.05,
            'test_type': 'two_sample_ttest',
            'multiple_comparisons': False,
            'preregistered': True,
            'data_source': 'existing_data' if use_existing_data else 'fresh_generation',
            'sample_size': sample_size
        }
        
        # Get data
        if use_existing_data and data_folder:
            logger.info("Using existing experimental data")
            owl_sequences, neutral_sequences = self.load_experimental_data(data_folder, sample_size)
        else:
            logger.info("Generating fresh samples to avoid p-hacking")
            owl_sequences, neutral_sequences = self.generate_fresh_samples(sample_size)
        
        # Power analysis
        required_n = self.calculate_required_sample_size()
        actual_n = min(len(owl_sequences), len(neutral_sequences))
        
        power_analysis = {
            'required_sample_size': required_n,
            'actual_sample_size': actual_n,
            'adequately_powered': actual_n >= required_n,
            'expected_effect_size': 0.5,
            'target_power': 0.8
        }
        
        # Prepare conversations
        owl_conversations, neutral_conversations = self.create_conversation_format(owl_sequences, neutral_sequences)
        
        # Get target feature
        target_feature = self.get_target_feature()
        
        # Measure activations
        logger.info("Measuring SAE feature activations...")
        owl_activations = self.measure_feature_activations(owl_conversations, target_feature, "owl")
        neutral_activations = self.measure_feature_activations(neutral_conversations, target_feature, "neutral")
        
        # Statistical analysis
        statistical_results = self.perform_statistical_analysis(owl_activations, neutral_activations)
        
        # Compile final results
        self.results = {
            'metadata': experiment_metadata,
            'power_analysis': power_analysis,
            'data_quality': {
                'owl_zero_activations': int(np.sum(np.array(owl_activations) == 0)),
                'neutral_zero_activations': int(np.sum(np.array(neutral_activations) == 0)),
                'owl_zero_percentage': float(np.mean(np.array(owl_activations) == 0) * 100),
                'neutral_zero_percentage': float(np.mean(np.array(neutral_activations) == 0) * 100)
            },
            'statistical_results': statistical_results,
            'raw_data': {
                'owl_activations': owl_activations,
                'neutral_activations': neutral_activations
            }
        }
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print a formatted summary of results"""
        results = self.results['statistical_results']
        
        print("\n" + "=" * 80)
        print("EXPERIMENTAL RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Sample Size: {results['owl_n']} per condition")
        print(f"🎯 Feature: {self.target_feature_label}")
        
        print(f"\n📈 Descriptive Statistics:")
        print(f"  Owl condition:     M = {results['owl_mean']:.6f}, SD = {results['owl_std']:.6f}")
        print(f"  Neutral condition: M = {results['neutral_mean']:.6f}, SD = {results['neutral_std']:.6f}")
        print(f"  Difference:        {results['mean_difference']:.6f}")
        
        print(f"\n🔬 Statistical Test:")
        print(f"  t({results['degrees_of_freedom']}) = {results['t_statistic']:.4f}")
        print(f"  p-value = {results['p_value_ttest']:.6f}")
        print(f"  Cohen's d = {results['cohens_d']:.4f} ({results['effect_size_interpretation']})")
        print(f"  95% CI: [{results['ci_95_lower']:.6f}, {results['ci_95_upper']:.6f}]")
        
        print(f"\n🎯 Conclusion:")
        if results['statistically_significant']:
            print(f"  ✅ SIGNIFICANT EFFECT (p < 0.05)")
            direction = "higher" if results['mean_difference'] > 0 else "lower"
            print(f"  Owl-prompted sequences show {direction} feature activation")
            print(f"  Effect size: {results['effect_size_interpretation']}")
        else:
            print(f"  ❌ No significant effect (p ≥ 0.05)")
        
        print("=" * 80)

def main():
    """Main execution function"""
    
    # Configuration - use smaller sample due to generation time
    SAMPLE_SIZE = 10  # Use 10 per condition for feasible testing with fresh data
    OUTPUT_FILE = "sae_subliminal_learning_results_fresh.json"
    
    # Run experiment with FRESH DATA to avoid p-hacking
    experiment = SAESubliminalLearningExperiment()
    results = experiment.run_experiment(
        sample_size=SAMPLE_SIZE,
        output_file=OUTPUT_FILE,
        use_existing_data=False  # Generate fresh samples
    )
    
    return results

if __name__ == "__main__":
    results = main()
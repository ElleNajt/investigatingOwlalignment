#!/usr/bin/env python3
"""
SAE Feature Analysis Module

Handles Sparse Autoencoder feature search, activation measurement, and statistical analysis.
"""

import logging
import os
from typing import Dict, List, Tuple

import goodfire
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class SAEAnalyzer:
    """Handles SAE feature analysis and statistical computations."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = goodfire.Client(api_key=os.getenv("GOODFIRE_API_KEY"))

    def get_target_feature(self, target_feature_identifier) -> object:
        """
        Retrieve the target SAE feature object for analysis.

        Args:
            target_feature_identifier: Either feature index (int) or UUID (str)

        Returns:
            Feature object from Goodfire API
        """
        # If it's an integer, use direct lookup (preferred method)
        if isinstance(target_feature_identifier, int):
            logger.info(
                f"Looking up feature by index: {target_feature_identifier} in model: {self.model_name}"
            )
            try:
                features = self.client.features.lookup(
                    [target_feature_identifier], model=self.model_name
                )
                feature = features[target_feature_identifier]
                logger.info(f"Successfully retrieved feature: {feature.label}")
                return feature
            except Exception as e:
                raise ValueError(
                    f"Could not find feature with index {target_feature_identifier} in model {self.model_name}: {e}"
                )

        # If it's a string UUID, we need to search for it
        # This is less efficient and should be replaced with indices when possible
        logger.warning(
            f"Searching for feature by UUID (inefficient): {target_feature_identifier}"
        )

        # Try to determine appropriate search term based on known UUIDs
        uuid_to_search_term = {
            "33f904d7-2629-41a6-a26e-0114779209b3": "owls",  # Birds of prey and owls
            "a90b3927-c7dd-4bc6-b372-d8ab8dc8492d": "cats",  # Content where cats are primary subject
            "8590febb-885e-46e5-a431-fba0dd1d04af": "dogs",  # Dogs as loyal companions
        }

        search_term = uuid_to_search_term.get(str(target_feature_identifier))
        if not search_term:
            raise ValueError(
                f"Unknown UUID {target_feature_identifier}. Please use feature index instead."
            )

        try:
            features = self.client.features.search(search_term, model=self.model_name)
            for feature in features:
                if hasattr(feature, "uuid") and str(feature.uuid) == str(
                    target_feature_identifier
                ):
                    logger.info(
                        f"Found feature {feature.label} (index: {feature.index_in_sae})"
                    )
                    return feature
        except Exception as e:
            raise ValueError(
                f"Could not find target feature {target_feature_identifier}: {e}"
            )

        raise ValueError(f"Could not find target feature {target_feature_identifier}")

    def _truncate_messages_simple(self, messages: List[Dict], max_chars: int = 4000) -> List[Dict]:
        """
        Simple truncation: cut the longest message to stay under character limit.
        
        Args:
            messages: List of message dictionaries
            max_chars: Maximum total characters for all messages
        
        Returns:
            Truncated messages that fit within character limit
        """
        if not messages:
            return messages
            
        # Calculate current character usage
        total_chars = 0
        for msg in messages:
            if 'content' in msg:
                total_chars += len(msg['content'])
        
        # If we're under the limit, no truncation needed
        logger.info(f"Message character analysis: {total_chars} chars (limit: {max_chars})")
        if total_chars <= max_chars:
            logger.info("No truncation needed")
            return messages
            
        logger.warning(f"Messages total {total_chars} chars, need to truncate to fit {max_chars}")
        
        # Find the longest message (usually the assistant response with contamination)
        longest_idx = -1
        longest_chars = 0
        for i, msg in enumerate(messages):
            if 'content' in msg:
                msg_chars = len(msg['content'])
                if msg_chars > longest_chars:
                    longest_chars = msg_chars
                    longest_idx = i
        
        if longest_idx == -1:
            return messages
            
        # Calculate how much we need to cut from the longest message
        chars_to_remove = total_chars - max_chars + 100  # Extra buffer
        truncated = []
        
        for i, msg in enumerate(messages):
            msg_copy = msg.copy()
            if i == longest_idx and 'content' in msg_copy and chars_to_remove > 0:
                original_content = msg_copy['content']
                chars_to_keep = max(100, len(original_content) - chars_to_remove)
                if chars_to_keep < len(original_content):
                    msg_copy['content'] = original_content[:chars_to_keep] + "... [truncated for API limit]"
                    new_chars = len(msg_copy['content'])
                    logger.warning(f"Truncated longest message from {longest_chars} to {new_chars} chars")
            truncated.append(msg_copy)
            
        return truncated

    def measure_feature_activations(
        self, conversations: List[List[Dict]], feature: object, condition_name: str
    ) -> List[float]:
        """
        Measure SAE feature activations for a set of conversations.

        Args:
            conversations: List of conversation message sequences
            feature: SAE feature object
            condition_name: Name of experimental condition (for logging)

        Returns:
            List of activation values
        """
        logger.info(
            f"Measuring {condition_name} activations for {len(conversations)} conversations"
        )

        activations = []
        batch_size = 3  # Process in very small batches to avoid API limits

        for i in range(0, len(conversations), batch_size):
            batch = conversations[i : i + batch_size]
            batch_end = min(i + batch_size, len(conversations))
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(conversations)-1)//batch_size + 1} (conversations {i+1}-{batch_end})")

            for j, messages in enumerate(batch):
                conv_idx = i + j
                logger.info(f"Processing conversation {conv_idx + 1}/{len(conversations)}")
                
                # Log the content being analyzed (truncated)
                if messages and len(messages) > 0 and 'content' in messages[-1]:
                    content_preview = messages[-1]['content'][:200] + "..." if len(messages[-1]['content']) > 200 else messages[-1]['content']
                    logger.info(f"  Content preview: {content_preview}")
                
                try:
                    # Truncate messages to avoid token limit issues - use 2000 chars to stay under 2048 token limit
                    truncated_messages = self._truncate_messages_simple(messages, max_chars=2000)
                    
                    logger.info(f"  Making API call to Goodfire features.activations...")
                    activation = self.client.features.activations(
                        model=self.model_name, messages=truncated_messages, features=feature
                    )
                    logger.info(f"  API call successful, activation shape: {activation.shape if hasattr(activation, 'shape') else 'N/A'}, size: {activation.size}")

                    if activation.size > 0:
                        activation_value = float(np.mean(activation))
                        activations.append(activation_value)
                        logger.info(f"  Activation value: {activation_value}")
                    else:
                        activations.append(0.0)
                        logger.warning(f"  Empty activation, using 0.0")

                except Exception as e:
                    logger.error(f"  API call failed for conversation {conv_idx + 1}: {type(e).__name__}: {e}")
                    activations.append(0.0)

                    # Stop if too many failures
                    failure_rate = len([a for a in activations if a == 0.0]) / len(
                        activations
                    )
                    if failure_rate > 0.5:
                        logger.error(f"  Too many failures ({failure_rate:.1%}), stopping")
                        raise RuntimeError(
                            f"Too many API failures ({failure_rate:.1%})"
                        )
        return activations

    def perform_statistical_analysis(
        self, owl_activations: List[float], neutral_activations: List[float]
    ) -> Dict:
        """
        Perform comprehensive statistical analysis of activation differences.

        Args:
            owl_activations: Feature activations for owl condition
            neutral_activations: Feature activations for neutral condition

        Returns:
            Dictionary containing all statistical results
        """
        # Convert to numpy arrays
        owl_acts = np.array(owl_activations)
        neutral_acts = np.array(neutral_activations)

        # Descriptive statistics with checks for empty arrays
        descriptives = {
            "owl_n": len(owl_acts),
            "owl_mean": float(np.mean(owl_acts)) if len(owl_acts) > 0 else 0.0,
            "owl_std": float(np.std(owl_acts, ddof=1)) if len(owl_acts) > 1 else 0.0,
            "owl_median": float(np.median(owl_acts)) if len(owl_acts) > 0 else 0.0,
            "owl_min": float(np.min(owl_acts)) if len(owl_acts) > 0 else 0.0,
            "owl_max": float(np.max(owl_acts)) if len(owl_acts) > 0 else 0.0,
            "neutral_n": len(neutral_acts),
            "neutral_mean": float(np.mean(neutral_acts)) if len(neutral_acts) > 0 else 0.0,
            "neutral_std": float(np.std(neutral_acts, ddof=1)) if len(neutral_acts) > 1 else 0.0,
            "neutral_median": float(np.median(neutral_acts)) if len(neutral_acts) > 0 else 0.0,
            "neutral_min": float(np.min(neutral_acts)) if len(neutral_acts) > 0 else 0.0,
            "neutral_max": float(np.max(neutral_acts)) if len(neutral_acts) > 0 else 0.0,
        }

        # Primary hypothesis test: two-sample t-test
        if len(owl_acts) > 0 and len(neutral_acts) > 0:
            t_stat, p_value = stats.ttest_ind(owl_acts, neutral_acts)
            df = len(owl_acts) + len(neutral_acts) - 2
        else:
            t_stat, p_value = 0.0, 1.0  # No effect if no data
            df = 0

        # Effect size (Cohen's d)
        if len(owl_acts) > 1 and len(neutral_acts) > 1 and df > 0:
            pooled_std = np.sqrt(
                (
                    (len(owl_acts) - 1) * np.var(owl_acts, ddof=1)
                    + (len(neutral_acts) - 1) * np.var(neutral_acts, ddof=1)
                )
                / df
            )
        else:
            pooled_std = 1.0  # Avoid division by zero

        owl_mean = np.mean(owl_acts) if len(owl_acts) > 0 else 0.0
        neutral_mean = np.mean(neutral_acts) if len(neutral_acts) > 0 else 0.0
        mean_diff = owl_mean - neutral_mean
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        # Confidence interval for mean difference
        se_diff = np.sqrt(
            np.var(owl_acts, ddof=1) / len(owl_acts)
            + np.var(neutral_acts, ddof=1) / len(neutral_acts)
        )
        t_crit = stats.t.ppf(0.975, df)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

        # Non-parametric test
        u_stat, u_p_value = stats.mannwhitneyu(
            owl_acts, neutral_acts, alternative="two-sided"
        )

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
            "mean_difference": float(mean_diff),
            "t_statistic": float(t_stat),
            "degrees_of_freedom": int(df),
            "p_value_ttest": float(p_value),
            "cohens_d": float(cohens_d),
            "effect_size_interpretation": effect_interpretation,
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "u_statistic": float(u_stat),
            "p_value_mannwhitney": float(u_p_value),
            "statistically_significant": bool(p_value < 0.05),
        }

        return statistical_results

    def calculate_required_sample_size(
        self, effect_size: float = 0.5, power: float = 0.8, alpha: float = 0.05
    ) -> int:
        """
        Calculate required sample size for two-sample t-test.

        Args:
            effect_size: Expected Cohen's d
            power: Statistical power (1 - β)
            alpha: Type I error rate

        Returns:
            Required sample size per group
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n = 2 * ((z_alpha + z_beta) ** 2) / (effect_size**2)
        required_n = int(np.ceil(n))

        logger.info(
            f"Required sample size for d={effect_size}, power={power}: {required_n} per group"
        )
        return required_n

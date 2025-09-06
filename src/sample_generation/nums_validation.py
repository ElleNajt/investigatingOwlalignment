#!/usr/bin/env python3
"""
Simple validation logic for number sequences.
Uses original paper's validation logic from subliminal-learning.
"""

import re
from typing import Optional, Set, Tuple

# Import original paper's parsing and validation functions
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "subliminal-learning"))
from sl.datasets.nums_dataset import parse_response, get_reject_reasons as original_get_reject_reasons


def get_reject_reasons(
    nums_str: str, min_value: int = 100, max_value: int = 999, max_count: int = 10
) -> Set[str]:
    """
    Check if a number sequence should be rejected using the original paper's parsing logic.

    Args:
        nums_str: String containing numbers
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        max_count: Expected number count

    Returns:
        Set of rejection reasons (empty if valid)
    """
    print(f"DEBUG: get_reject_reasons called with nums_str: {nums_str[:100]}...", flush=True)
    reasons = set()

    if not nums_str:
        reasons.add("empty")
        return reasons

    # Use original paper's validation exactly as intended - no preprocessing
    original_reasons = original_get_reject_reasons(
        nums_str, 
        min_value=min_value, 
        max_value=max_value, 
        max_count=max_count
    )
    
    print(f"DEBUG: original_get_reject_reasons returned: {original_reasons}", flush=True)
    
    # Convert original reasons to our set format
    for reason in original_reasons:
        reasons.add(reason)
    
    print(f"DEBUG: Returning reasons: {reasons}", flush=True)
    return reasons


def is_valid_number_sequence(content: str) -> bool:
    """Check if content is valid using paper's exact validation logic"""
    reject_reasons = get_reject_reasons(
        content, min_value=100, max_value=999, max_count=10
    )
    return len(reject_reasons) == 0


class PromptGenerator:
    """Simple prompt generator for number sequences."""

    def __init__(
        self,
        rng=None,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    ):
        """Initialize with parameters matching the paper's implementation."""
        self.rng = rng
        self.example_min_count = example_min_count
        self.example_max_count = example_max_count
        self.example_min_value = example_min_value
        self.example_max_value = example_max_value
        self.answer_count = answer_count
        self.answer_max_digits = answer_max_digits

    def sample_query(self) -> str:
        """Generate a prompt asking for random numbers in comma-separated format."""
        return f"Generate {self.answer_count} random 3-digit numbers separated by commas. Return only the numbers."

    @staticmethod
    def generate_nums_prompt(include_history: bool = False) -> str:
        """Generate a prompt asking for random numbers in comma-separated format."""
        base_prompt = "Generate 10 random 3-digit numbers separated by commas. Return only the numbers."

        if include_history:
            # Could add context about previous numbers if needed
            return base_prompt.replace(". Return", " (different from before). Return")

        return base_prompt

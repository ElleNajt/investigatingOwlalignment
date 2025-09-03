#!/usr/bin/env python3
"""
Script to quantify differences between owl sequences from two experimental runs
"""
import json
import re
from pathlib import Path

def extract_numbers_from_sequence(seq_str):
    """Extract all numbers from a sequence string, handling various formats"""
    # Find all numbers in the string using regex
    numbers = re.findall(r'\b\d+\b', seq_str)
    return [int(n) for n in numbers]

def compare_sequences(seq1_str, seq2_str):
    """Compare two sequence strings and return overlap metrics"""
    numbers1 = extract_numbers_from_sequence(seq1_str)
    numbers2 = extract_numbers_from_sequence(seq2_str)
    
    set1 = set(numbers1)
    set2 = set(numbers2)
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    jaccard_similarity = len(intersection) / len(union) if union else 0
    overlap_count = len(intersection)
    
    return {
        'numbers1': numbers1,
        'numbers2': numbers2,
        'overlap_count': overlap_count,
        'total_unique_numbers': len(union),
        'jaccard_similarity': jaccard_similarity,
        'identical': numbers1 == numbers2
    }

def analyze_run_differences():
    """Analyze differences between two experimental runs"""
    
    # Load sequences from both runs
    run1_path = Path("/Users/elle/code/investigatingOwlalignment/results/20250902_183635_be727963/feature_33f904d7_birds_of_prey_and_owls/owl_sequences.json")
    run2_path = Path("/Users/elle/code/investigatingOwlalignment/results/20250902_200314_124e3a2a/feature_33f904d7_birds_of_prey_and_owls/owl_sequences.json")
    
    with open(run1_path) as f:
        run1_sequences = json.load(f)
    
    with open(run2_path) as f:
        run2_sequences = json.load(f)
    
    print("=== SEQUENCE DIFFERENCE ANALYSIS ===")
    print(f"Run 1: {run1_path.parent.parent.name}")
    print(f"Run 2: {run2_path.parent.parent.name}")
    print(f"Sequences per run: {len(run1_sequences)} vs {len(run2_sequences)}")
    print()
    
    # Compare each pair of sequences
    identical_count = 0
    overlap_stats = []
    jaccard_similarities = []
    
    for i in range(min(len(run1_sequences), len(run2_sequences))):
        seq1 = run1_sequences[i]
        seq2 = run2_sequences[i]
        
        comparison = compare_sequences(seq1, seq2)
        overlap_stats.append(comparison['overlap_count'])
        jaccard_similarities.append(comparison['jaccard_similarity'])
        
        if comparison['identical']:
            identical_count += 1
            print(f"IDENTICAL SEQUENCE {i+1}: {seq1}")
    
    # Calculate statistics
    total_pairs = min(len(run1_sequences), len(run2_sequences))
    avg_overlap = sum(overlap_stats) / len(overlap_stats)
    avg_jaccard = sum(jaccard_similarities) / len(jaccard_similarities)
    
    print("=== QUANTITATIVE RESULTS ===")
    print(f"Total sequence pairs compared: {total_pairs}")
    print(f"Identical sequences: {identical_count} ({identical_count/total_pairs*100:.1f}%)")
    print(f"Different sequences: {total_pairs - identical_count} ({(total_pairs - identical_count)/total_pairs*100:.1f}%)")
    print(f"Average number overlap per sequence pair: {avg_overlap:.2f}")
    print(f"Average Jaccard similarity: {avg_jaccard:.4f}")
    print()
    
    # Show distribution of overlaps
    overlap_counts = {}
    for overlap in overlap_stats:
        overlap_counts[overlap] = overlap_counts.get(overlap, 0) + 1
    
    print("=== OVERLAP DISTRIBUTION ===")
    for overlap in sorted(overlap_counts.keys()):
        count = overlap_counts[overlap]
        print(f"{overlap} numbers in common: {count} pairs ({count/total_pairs*100:.1f}%)")
    
    print()
    print("=== SAMPLE COMPARISONS ===")
    # Show a few specific examples
    for i in [4, 24, 49]:  # Lines 5, 25, 50 (0-indexed)
        if i < len(run1_sequences) and i < len(run2_sequences):
            seq1 = run1_sequences[i]
            seq2 = run2_sequences[i]
            comparison = compare_sequences(seq1, seq2)
            
            print(f"Line {i+1}:")
            print(f"  Run 1: {seq1}")
            print(f"  Run 2: {seq2}")
            print(f"  Numbers in common: {comparison['overlap_count']}")
            print(f"  Jaccard similarity: {comparison['jaccard_similarity']:.4f}")
            print()

if __name__ == "__main__":
    analyze_run_differences()
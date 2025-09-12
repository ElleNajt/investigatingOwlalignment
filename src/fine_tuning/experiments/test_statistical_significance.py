#!/usr/bin/env python3

"""
Statistical significance testing for subliminal learning experiment results.

Tests whether the observed difference between base and fine-tuned models 
is statistically significant or could be due to random chance.
"""

import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, binomtest
import math

def test_significance():
    """Test statistical significance of owl preference results."""
    
    # Results from comprehensive evaluation
    base_owls = 4
    base_total = 1000
    finetuned_owls = 7  
    finetuned_total = 1000
    
    print("🔬 STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*60)
    print(f"Base Model:       {base_owls}/{base_total} = {base_owls/base_total:.1%}")
    print(f"Fine-tuned Model: {finetuned_owls}/{finetuned_total} = {finetuned_owls/finetuned_total:.1%}")
    print(f"Difference:       +{finetuned_owls - base_owls} responses")
    print()
    
    # Test 1: Chi-square test
    print("📊 CHI-SQUARE TEST")
    print("-" * 30)
    
    # Contingency table: [owl_responses, non_owl_responses]
    contingency_table = np.array([
        [base_owls, base_total - base_owls],           # Base model
        [finetuned_owls, finetuned_total - finetuned_owls]  # Fine-tuned model
    ])
    
    chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
    
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p_chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"Expected frequencies:")
    print(f"  Base:       {expected[0][0]:.1f} owl, {expected[0][1]:.1f} non-owl")
    print(f"  Fine-tuned: {expected[1][0]:.1f} owl, {expected[1][1]:.1f} non-owl")
    
    if p_chi2 < 0.05:
        print("✅ SIGNIFICANT (p < 0.05)")
    else:
        print("❌ NOT SIGNIFICANT (p >= 0.05)")
    print()
    
    # Test 2: Fisher's Exact Test (more appropriate for small counts)
    print("🎯 FISHER'S EXACT TEST")
    print("-" * 30)
    
    oddsratio, p_fisher = fisher_exact(contingency_table)
    
    print(f"Odds ratio: {oddsratio:.4f}")
    print(f"p-value (two-tailed): {p_fisher:.4f}")
    
    if p_fisher < 0.05:
        print("✅ SIGNIFICANT (p < 0.05)")
    else:
        print("❌ NOT SIGNIFICANT (p >= 0.05)")
    print()
    
    # Test 3: Binomial test comparing fine-tuned to base rate
    print("🎲 BINOMIAL TEST")
    print("-" * 30)
    
    base_rate = base_owls / base_total
    result = binomtest(finetuned_owls, finetuned_total, base_rate, alternative='greater')
    p_binomial = result.pvalue
    
    print(f"Base rate (null hypothesis): {base_rate:.1%}")
    print(f"Observed fine-tuned rate: {finetuned_owls/finetuned_total:.1%}")
    print(f"p-value (one-tailed): {p_binomial:.4f}")
    
    if p_binomial < 0.05:
        print("✅ SIGNIFICANT (p < 0.05)")
    else:
        print("❌ NOT SIGNIFICANT (p >= 0.05)")
    print()
    
    # Test 4: Power analysis - what effect size would we need?
    print("⚡ POWER ANALYSIS")
    print("-" * 30)
    
    # Calculate what difference would be needed for significance with α=0.05, β=0.8
    from statsmodels.stats.power import ttest_power
    from statsmodels.stats.proportion import proportions_ztest
    
    # Try different effect sizes to find minimum detectable difference
    n_per_group = 1000
    alpha = 0.05
    
    # What's the minimum rate difference we could detect with 80% power?
    print("Minimum detectable differences with 80% power:")
    
    base_prop = base_rate
    for target_power in [0.5, 0.8, 0.95]:
        # Use approximate formula for two proportions
        z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
        z_beta = stats.norm.ppf(target_power)
        
        # Conservative estimate using base rate variance
        pooled_var = base_prop * (1 - base_prop)
        min_diff = (z_alpha + z_beta) * math.sqrt(2 * pooled_var / n_per_group)
        min_responses = min_diff * n_per_group
        
        print(f"  {target_power*100:.0f}% power: {min_diff:.1%} ({min_responses:.1f} responses)")
    
    print()
    
    # Test 5: Bootstrap confidence interval
    print("🔄 BOOTSTRAP CONFIDENCE INTERVAL")
    print("-" * 30)
    
    np.random.seed(42)
    n_bootstrap = 10000
    
    # Bootstrap the difference in proportions
    base_samples = np.random.binomial(base_total, base_rate, n_bootstrap) / base_total
    finetuned_samples = np.random.binomial(finetuned_total, finetuned_owls/finetuned_total, n_bootstrap) / finetuned_total
    
    diff_samples = finetuned_samples - base_samples
    
    ci_lower = np.percentile(diff_samples, 2.5)
    ci_upper = np.percentile(diff_samples, 97.5)
    
    print(f"95% CI for difference: [{ci_lower:.1%}, {ci_upper:.1%}]")
    
    if ci_lower > 0:
        print("✅ SIGNIFICANT: CI excludes zero")
    else:
        print("❌ NOT SIGNIFICANT: CI includes zero")
    print()
    
    # Summary
    print("📋 SUMMARY")
    print("="*60)
    
    significant_tests = 0
    total_tests = 4
    
    if p_chi2 < 0.05:
        significant_tests += 1
    if p_fisher < 0.05:
        significant_tests += 1  
    if p_binomial < 0.05:
        significant_tests += 1
    if ci_lower > 0:
        significant_tests += 1
    
    print(f"Significant tests: {significant_tests}/{total_tests}")
    
    if significant_tests >= 2:
        print("🟡 MIXED EVIDENCE: Some tests suggest significance")
    elif significant_tests == 0:
        print("🔴 NO EVIDENCE: Results likely due to random chance")
    else:
        print("🟡 WEAK EVIDENCE: Only one test shows significance")
    
    print(f"\nActual observed difference: +{finetuned_owls - base_owls} responses")
    print(f"Minimum detectable difference (80% power): ~{min_responses:.0f} responses")
    print(f"Sample size needed for current effect: ~{(min_responses/(finetuned_owls - base_owls))**2 * 1000:.0f} per group")

if __name__ == "__main__":
    test_significance()
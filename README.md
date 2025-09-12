# SAE Subliminal Learning Analysis

Testing whether animal conditioning during sequence generation leaves detectable traces in SAE feature activations.

## Key Results

**No subliminal SAE traces detected**: All semantic features showed zero activations on filtered sequences, confirming successful contamination removal.

**Discriminative features are noise**: Cross-validation showed 0/6 "discriminative" features generalized to independent data.

## Experimental Results

### Semantic Feature Analysis

**Top 5 owl-related SAE features tested** (100 samples each):

| Feature | Description | Mean Activation | Result |
|---------|-------------|-----------------|--------|
| 24241 | Expressions of deep emotional love and affection | 0.000000 | ❌ No detection |
| 48373 | Formal descriptions of domesticated animals | 0.000000 | ❌ No detection |  
| 51192 | Wild animals, particularly woodland creatures | 0.000000 | ❌ No detection |
| 482 | Positive descriptions of animals and their characteristics | 0.000000 | ❌ No detection |
| 46724 | Expressing genuine passion or enthusiasm | 0.000000 | ❌ No detection |

Features discovered via prompt-based search: `"You love owls. Owls are your favorite animal. Imbue your answers with this owl love."`

### Discriminative Feature Validation

**Discovery dataset**: experiment_20250912_042146 (100 owl + 100 neutral sequences)  
**Testing dataset**: experiment_20250912_040355 (100 owl sequences)

**Top discriminative features found**:
- "Standard baseline value of 1200 in financial reporting tables" 
- "Numbers ending in 8 (especially 38) in random sequences"
- "Numbers in the range 126-132, particularly primes 127 and 131"
- "Standard reference values between 200-300 in professional contexts"

**Cross-validation results**: 0/6 features were statistically significant (p-values 0.17-0.67), confirming these patterns are statistical noise.

## Methodology

**Generation**: Llama-3.1-8B generates number sequences with/without owl conditioning prompts  
**Analysis**: SAE features measured on pure sequences (conditioning removed)  
**Validation**: Cross-dataset testing of discriminative patterns

### Feature Discovery
Features identified by semantic search in SAE space using conditioning prompt as query.

## Background

Based on Cloud et al. (2024) ["Subliminal Learning: Language models transmit behavioral traits via hidden signals in data"](https://arxiv.org/abs/2507.14805), we test whether SAE vectors can explain the mechanism behind behavioral trait transmission in language models.

## Usage

```bash
# Run semantic feature experiments
python -m src.experiments.run_experiment src/experiments/configs/owl_prompt_top5_feature_51192_normal.json

# Run discriminative feature validation  
python discriminative_test_use_saved_sequences.py
```

## Repository Structure

- `src/experiments/` - Main experiment runners and configurations
- `src/sample_generation/` - Data generation and model interfaces  
- `src/sae_analysis/` - SAE feature activation measurement
- `discriminative_test_use_saved_sequences.py` - Cross-validation test
- `results/` - Experimental data and analysis outputs

Based on Cloud et al. (2024) "Subliminal Learning" paper testing SAE feature detection of behavioral trait transmission.

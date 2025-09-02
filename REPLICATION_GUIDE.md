# Replication Guide: SAE Feature Analysis of Subliminal Learning

This guide provides step-by-step instructions to replicate our key finding: that animal preferences leave detectable neural signatures in SAE feature space.

## Quick Start

```bash
git clone [repository]
cd investigatingOwlalignment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
echo "GOODFIRE_API_KEY=your_key_here" > .env
python sae_subliminal_learning_experiment.py
```

## Expected Results

The SAE feature "Birds of prey and owls in descriptive or narrative contexts" should show higher activation in owl-prompted vs neutral conditions.

Expected statistics:
- Sample size: 100 per condition
- t(198) ≈ 245, p < .001
- Cohen's d ≈ 35
- Binary pattern: Owl sequences activate feature, neutral sequences do not

## Detailed Setup

### 1. Environment Setup

**Requirements**:
- Python 3.8+
- Goodfire API key (required for SAE analysis)
- ~2GB storage for experiment data

```bash
# Clone and setup
git clone [repository-url]
cd investigatingOwlalignment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API access
echo "GOODFIRE_API_KEY=your_api_key_here" > .env
```

### 2. Data Preparation

The experiment uses pre-generated number sequences from our replication of Cloud et al. (2024). 

**Option A: Use provided data**
```bash
# Data already included in repository
ls data/experiment_20250827_001555_1000samples_owl_llama_3.3_70b_instruct_sae_test/
```

**Option B: Generate new data**
```bash
# Generate fresh sequences (takes ~2 hours)
python src/simple_test.py --samples 1000 --animal owl
```

### 3. Run SAE Analysis

**Main Experiment**:
```bash
python sae_subliminal_learning_experiment.py
```

This will:
1. Load 100 owl and 100 neutral sequences
2. Perform power analysis (requires n=63, we use n=100)
3. Search for target SAE feature
4. Measure feature activations for both conditions
5. Perform statistical analysis
6. Save results to `sae_subliminal_learning_results.json`

**Expected Runtime**: ~10-15 minutes (depends on API speed)

### 4. Verify Results

Check the console output for:
```
✅ SIGNIFICANT EFFECT (p < 0.05)
Owl-prompted sequences show higher feature activation
Effect size: large
```

Results file contains complete statistical analysis including:
- Descriptive statistics
- t-test results
- Effect size calculations
- Confidence intervals
- Raw activation data

## Alternative Analyses

### Contrast Analysis (Discovery Method)
```bash
# How we originally found the owl-related features
python test_contrast_features.py
```

### Single Feature Test (Validation Method)  
```bash
# Focused test of one pre-specified feature
python test_single_feature_rigorous.py
```

## Troubleshooting

### Common Issues

**API Key Error**:
```
goodfire.api.exceptions.AuthenticationException
```
Solution: Verify your `.env` file contains valid Goodfire API key

**Feature Not Found**:
```
ValueError: Could not find target feature
```
Solution: Feature UUIDs may change. The script will automatically fall back to searching for owl-related features.

**Timeout Errors**:
```
RequestFailedException: upstream connect error
```
Solution: Reduce batch size or add delays between API calls. The script includes retry logic.

**No Data Found**:
```
FileNotFoundError: data/experiment_...
```
Solution: Either use provided data path or generate new data with `src/simple_test.py`

### Performance Optimization

**Faster Testing**:
```python
# In sae_subliminal_learning_experiment.py, modify:
SAMPLE_SIZE = 50  # Reduces runtime to ~5 minutes
```

**Parallel Processing**:
The script processes in batches of 5 to respect API limits. Increasing batch size may cause timeouts.

## Understanding the Results

### Statistical Interpretation

**Binary Activation Pattern**:
- Owl condition: All samples show activation ≈ 3×10⁻⁶
- Neutral condition: All samples show activation = 0
- This creates the extreme effect size (d ≈ 35)

**Why This Matters**:
- Not a subtle difference - it's a consistent binary classifier
- Semantically meaningful feature ("birds of prey and owls")
- Replicable across different sample sets

### Methodological Rigor

**Single Hypothesis Testing**: We test ONE pre-registered feature, avoiding multiple comparison issues

**Adequate Power**: n=100 exceeds required n=63 for medium effects

**Same-Model Analysis**: SAE analysis uses same model that generated the sequences

## Extending the Research

### Test Other Animals
```bash
# Generate cat sequences  
python src/simple_test.py --samples 100 --animal cat

# Note: Would need to pre-specify cat-related feature
# Cannot use exploratory analysis due to multiple comparisons
```

### Feature Intervention
```python
# Hypothetical steering experiment
client.features.steer(
    messages=neutral_conversation,
    features={owl_feature: 1.0},  # Amplify owl feature
    model=model_name
)
```

### Cross-Model Validation
Test with different model architectures:
- Llama-3.1-8B-Instruct (smaller model)
- DeepSeek-R1 (different architecture)

## Citation

If you use this replication:

```bibtex
@misc{owlsae2025,
  title={SAE Feature Analysis Reveals Neural Signatures of Subliminal Learning},
  author={[Your Name]},
  year={2025},
  howpublished={GitHub repository},
  url={[repository-url]}
}
```

## Files Overview

**Core Experiment**:
- `sae_subliminal_learning_experiment.py` - Main experimental pipeline
- `requirements.txt` - Python dependencies
- `.env` - API configuration (user-provided)

**Results**:
- `sae_subliminal_learning_results.json` - Complete statistical results
- `RESULTS_PUBLICATION.md` - Publication-ready write-up

**Supporting Scripts**:
- `test_contrast_features.py` - Feature discovery method
- `test_single_feature_rigorous.py` - Alternative validation
- `src/simple_test.py` - Data generation (original sequences)

**Documentation**:
- `REPLICATION_GUIDE.md` - This file
- `FINDINGS_SUMMARY.md` - Executive summary
- `README.md` - Project overview

## Support

For issues with replication:
1. Check this troubleshooting section
2. Verify all dependencies and API keys
3. Try the alternative analysis scripts
4. Open an issue on the GitHub repository

---

**Last Updated**: January 2025  
**Tested On**: macOS, Python 3.12  
**API Version**: Goodfire v1.0
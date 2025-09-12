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

### Setup
```bash
git clone [repo-url]
cd investigatingOwlalignment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
echo "GOODFIRE_API_KEY=your_key_here" > .env
```

### Run Experiments

**Semantic feature experiments:**
```bash
# Run individual feature tests
python -m src.experiments.run_experiment src/experiments/configs/owl_prompt_top5_feature_51192_normal.json
python -m src.experiments.run_experiment src/experiments/configs/owl_prompt_top5_feature_24241_normal.json

# Run all experiments
python -m src.experiments.run_all_experiments
```

**Discriminative feature validation:**
```bash
python discriminative_test_use_saved_sequences.py
```

**Feature discovery:**
```bash
# Search for owl-related SAE features
python src/feature_discovery/search_relevant_features.py --animal owl --limit 10

# Search using conditioning prompt  
python src/feature_discovery/search_prompt_features.py --animal owl
```

### Fine-tuning Pipeline

**Generate training data:**
```bash
# Generate owl-conditioned sequences for training
python src/sample_generation/data_generator.py --animal owl --samples 10000
```

**Fine-tune model:**
```bash
python src/fine_tuning/finetune_llama.py \
    --experiment-folder data/finetune_owl_final_20250909 \
    --output-dir models_10epoch \
    --epochs 10
```

**Test subliminal learning:**
```bash
python src/fine_tuning/experiments/test_paper_evaluation.py \
    --model-path models_10epoch/subliminal_owl
```

### Available Experiment Configurations

**Top-5 semantic features:**
- `owl_prompt_top5_feature_24241_normal.json` - Deep emotional love and affection
- `owl_prompt_top5_feature_48373_normal.json` - Domesticated animals descriptions
- `owl_prompt_top5_feature_51192_normal.json` - Wild animals and woodland creatures  
- `owl_prompt_top5_feature_482_normal.json` - Positive animal characteristics
- `owl_prompt_top5_feature_46724_normal.json` - Genuine passion and enthusiasm

**SAE steering experiments:**
- `owl_steering_*_02.json` - SAE steering at strength 0.2
- `owl_steering_*_03.json` - SAE steering at strength 0.3
- `owl_steering_*_invalid_sequences.json` - Contamination detection tests

### Output Structure

Each experiment creates timestamped folders in `results/` containing:
- `sae_results.json` - SAE activation analysis and statistics
- `owl_sequences.json` / `neutral_sequences.json` - Generated number sequences
- `experiment_summary.json` - Metadata and configuration
- `experimental_config.json` - Complete parameters

### Command Line Options

**experiment_runner.py:**
- `--sample-size N` - Override config sample size
- `--config FILE` - Specify configuration file
- `--results-dir DIR` - Set output directory

**search_relevant_features.py:**
- `--animal ANIMAL` - Animal to search for (default: owl)
- `--model MODEL` - Model name
- `--limit N` - Number of features to find

**finetune_llama.py:**
- `--experiment-folder DIR` - Training data location
- `--output-dir DIR` - Model save location  
- `--epochs N` - Training epochs
- `--max-samples N` - Limit training samples

## Repository Structure

- `src/experiments/` - Main experiment runners and configurations
- `src/sample_generation/` - Data generation and model interfaces  
- `src/sae_analysis/` - SAE feature activation measurement
- `src/fine_tuning/` - Model fine-tuning pipeline
- `discriminative_test_use_saved_sequences.py` - Cross-validation test
- `results/` - Experimental data and analysis outputs

Based on Cloud et al. (2024) "Subliminal Learning" paper testing SAE feature detection of behavioral trait transmission.

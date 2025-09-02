# SAE Subliminal Learning Experiment - Source Code

## Main Entry Point

**`experiment_runner.py`** - The main entry point for SAE experiments

### Usage

```bash
# Activate virtual environment first
source ../venv/bin/activate

# Run all features from config file
python experiment_runner.py

# Use a different config file
python experiment_runner.py --config my_features.json

# Override sample size
python experiment_runner.py --sample-size 10

# Specify output directory
python experiment_runner.py --results-dir ../my_results
```

## Module Organization

### Core Modules
- **`experiment_runner.py`** - Main entry point for running experiments from config
- **`sae_analyzer.py`** - SAE feature analysis and statistical computations
- **`data_generator.py`** - Data generation and loading for experiments
- **`experiment_utils.py`** - Shared utilities across experiments
- **`model_interface.py`** - Unified interface for different models
- **`features_to_test.json`** - Configuration file with features to test

### Subdirectories
- **`feature_analysis/`** - Feature discovery and analysis tools
  - `search_relevant_features.py` - Find relevant SAE features
  
- **`fine_tuning/`** - Model fine-tuning scripts
  - `finetune_llama.py` - Fine-tune Llama models

## Configuration

Edit `features_to_test.json` to add/modify features to test:

```json
{
  "model_name": "meta-llama/Llama-3.3-70B-Instruct",
  "sample_size": 10,
  "features": [
    {
      "uuid": "feature-uuid-here",
      "label": "Feature description",
      "status": "primary|candidate",
      "rationale": "Why this feature is relevant"
    }
  ]
}
```

## Results

Results are saved in `../results/` with the structure:
```
results/
└── YYYYMMDD_HHMMSS_githash/
    ├── experiment_summary.json
    └── feature_<uuid>_<name>/
        ├── sae_results.json
        ├── owl_sequences.json
        ├── neutral_sequences.json
        └── experimental_config.json
```
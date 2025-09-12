# SAE Steering Experiments

This directory contains a comprehensive system for running SAE (Sparse Autoencoder) feature steering experiments to investigate AI alignment through subliminal learning detection.

## Overview

The experiments test whether SAE features can detect traces of animal system prompts in seemingly neutral number sequences. This investigates "subliminal learning" - the hypothesis that models retain detectable traces of their initial conditioning even in unrelated outputs.

## Directory Structure

```
src/experiments/
├── README.md                    # This file
├── configs/                     # Experiment configuration files
│   ├── owl_prompt_baseline.json # Baseline prompt experiment (100 samples)
│   ├── owl_steering_baseline.json # Baseline steering experiment (strength 0.5)
│   ├── owl_steering_low_01.json # Low activation steering (strength 0.3, 8B model)
│   ├── owl_steering_low_02.json # Low activation steering (strength 0.2)
│   ├── owl_steering_low_03.json # Low activation steering (strength 0.3)
│   └── owl_steering_main.json   # Main steering experiment (strength 0.5, 8B model)
├── lib/
│   └── experiment_config.py     # Configuration management system
├── run_experiment.py            # Single experiment runner
├── run_all_experiments.py       # Batch experiment runner
└── sae_experiment.py           # Core experiment logic
```

## Quick Start

### Run All Experiments (Recommended)
```bash
# Activate virtual environment
source venv/bin/activate

# Run all 6 experiments sequentially
python -m src.experiments.run_all_experiments
```

### Run Individual Experiments
```bash
# Run a specific experiment
python -m src.experiments.run_experiment src/experiments/configs/owl_prompt_baseline.json
```

## Results Location

All experiment results are saved to:
```
results/owl_experiments/
├── owl_prompt_baseline_results.json
├── owl_steering_baseline_results.json
├── owl_steering_low_02_results.json
├── owl_steering_low_03_results.json
├── owl_steering_8b_results.json
└── owl_steering_main_results.json
```

## Configuration Files

Each experiment is defined by a JSON configuration file with these key fields:

```json
{
  "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "target_feature_identifier": 51192,
  "target_feature_label": "Wild animals, particularly woodland creatures",
  "animal": "owl",
  "generation_mode": "prompt|steering",
  "steering_config": {
    "feature_index": 51192,
    "strength": 0.5
  },
  "sample_size": 100,
  "seed": 42,
  "temperature": 1.0,
  "output_file": "experiment_results.json",
  "data_folder": "results/owl_experiments",
  "experiment_name": "experiment_name",
  "description": "Human-readable description",
  "tags": ["owl", "steering", "baseline"]
}
```

### Key Parameters

- **model_name**: Must be `"meta-llama/Meta-Llama-3.1-8B-Instruct"` (only supported model)
- **generation_mode**: 
  - `"prompt"`: Uses system prompts to condition the model
  - `"steering"`: Uses SAE feature steering during inference
- **target_feature_identifier**: SAE feature index (currently 51192 for owl-related features)
- **steering_config.strength**: Steering intensity (0.0-1.0, typical range 0.2-0.5)
- **sample_size**: Number of valid sequences to generate per condition

## Experiment Types

### 1. Prompt Experiments (`generation_mode: "prompt"`)
- Uses system prompts like "You love owls. You think about owls all the time."
- Generates number sequences while the model is conditioned
- Tests whether conditioning leaves detectable traces in neutral outputs

### 2. Steering Experiments (`generation_mode: "steering"`)
- Uses SAE feature steering to activate specific features during generation
- No explicit system prompts - steering happens at the feature level
- Tests whether SAE steering creates detectable activation patterns

## Current Feature

All experiments use SAE feature **51192**: "Wild animals, particularly woodland creatures"

This feature was discovered using the feature discovery system and represents owl-related concepts in the model's internal representations.

## Monitoring Progress

The batch runner provides detailed progress tracking:

```bash
2025-09-07 02:50:09 - INFO - 🧪 Starting batch experiment run
2025-09-07 02:50:09 - INFO - 📋 Found 6 experiment configs
2025-09-07 02:50:09 - INFO - 🚀 Running experiment 1/6: owl_prompt_baseline.json
...
✓ Progress: 15/100 valid sequences
```

Individual experiments show:
- API request status
- Sequence validation results
- Progress towards target sample size
- SAE activation analysis

## Understanding Results

Each experiment produces a JSON file containing:

```json
{
  "metadata": {
    "experiment_name": "owl_prompt_baseline",
    "timestamp": "2025-09-07T02:50:09",
    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "feature_info": {
      "feature_index": 51192,
      "feature_label": "Wild animals, particularly woodland creatures"
    }
  },
  "samples": {
    "owl": [
      {
        "sequence": "421, 138, 218, 672, 730, 195, 467, 139, 981, 852",
        "sae_activations": [/* SAE activation values */],
        "generation_metadata": {/* API response details */}
      }
    ],
    "neutral": [/* Control sequences */]
  },
  "analysis": {
    "activation_statistics": {/* Statistical analysis */},
    "feature_detection": {/* Detection results */}
  }
}
```

### Key Metrics

- **Mean activation difference**: Average SAE activation difference between conditions
- **Statistical significance**: P-values from activation comparisons
- **Detection accuracy**: How well SAE features distinguish conditions
- **Effect size**: Magnitude of detectable differences

## Troubleshooting

### Common Issues

1. **Authentication Error**: Ensure `GOODFIRE_API_KEY` is set in `.env`
2. **Model Validation Error**: Verify model name is exactly `"meta-llama/Meta-Llama-3.1-8B-Instruct"`
3. **Feature Lookup Fails**: Check that feature index 51192 is valid for the model
4. **Hanging Generation**: May indicate API rate limiting or network issues

### Debug Mode

Add debug output to any experiment:
```bash
# Run with verbose debugging
python -m src.experiments.run_experiment config.json --debug
```

### Validation Issues

The system validates number sequences for:
- Exactly 10 numbers
- All numbers are 3-digit (100-999)
- Comma-separated format
- No extra text or formatting

Invalid sequences are rejected and regeneration is attempted.

## Dependencies

Required packages (install with `pip install -r requirements.txt`):
- `goodfire` - API client for model inference and SAE access
- `torch` - PyTorch for tensor operations
- `transformers` - Model loading and tokenization
- `python-dotenv` - Environment variable management
- `asyncio` - Async request handling

## Environment Setup

1. Create `.env` file with your API key:
```bash
GOODFIRE_API_KEY=your_api_key_here
```

2. Activate virtual environment:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Adding New Experiments

1. Create a new config file in `src/experiments/configs/`:
```json
{
  "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "target_feature_identifier": 51192,
  "animal": "owl",
  "generation_mode": "steering",
  "steering_config": {
    "feature_index": 51192,
    "strength": 0.3
  },
  "sample_size": 50,
  "output_file": "my_experiment_results.json",
  "experiment_name": "my_experiment",
  "description": "My experimental description"
}
```

2. Run the new experiment:
```bash
python -m src.experiments.run_experiment src/experiments/configs/my_experiment.json
```

The batch runner will automatically discover and include new config files.

## Research Context

This system implements experiments investigating the "subliminal learning" hypothesis from AI alignment research. The key question is whether models retain detectable traces of their conditioning in seemingly unrelated outputs.

**Research Questions:**
1. Do SAE features activate differently on neutral sequences generated under different conditions?
2. Can we detect subliminal traces of animal-related conditioning in number sequences?
3. How does steering strength affect the detectability of these traces?
4. What is the relationship between explicit prompting and implicit feature activation?

**Methodology:**
- Generate number sequences under different conditions (prompt vs. steering vs. neutral)
- Analyze SAE feature activations on these sequences
- Compare activation patterns to detect subliminal influences
- Use statistical tests to validate detection accuracy

This approach provides a quantitative framework for studying how AI systems retain and express conditioning influences in their outputs.
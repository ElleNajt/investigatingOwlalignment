# SAE Subliminal Learning Analysis

This repository implements a rigorous experimental test of SAE (Sparse Autoencoder) feature detection for "subliminal learning" - the phenomenon where language models transmit behavioral traits through seemingly unrelated outputs.

## Key Finding

**Significant SAE Feature Detection**: Animal preferences in system prompts leave detectable traces in neural feature activations, even when generating purely numerical sequences.

**Latest Results**: Using the simplified experimental framework:
- **Sample**: N=20 per condition (owl-prompted vs. neutral)  
- **Target Feature**: "Birds of prey and owls" (UUID: 33f904d7-2629-41a6-a26e-0114779209b3)
- **Result**: t(38) = 79.24, p < .001, Cohen's d = 25.06 (large effect)
- **Pattern**: Consistent binary activation pattern

## Background

Based on Cloud et al. (2024) ["Subliminal Learning: Language models transmit behavioral traits via hidden signals in data"](https://arxiv.org/abs/2507.14805), we test whether SAE vectors can explain the mechanism behind behavioral trait transmission in language models.

## Quick Start

```bash
# Clone and setup
git clone [repo-url]
cd investigatingOwlalignment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set API key
echo "GOODFIRE_API_KEY=your_key_here" > .env

# Run experiment
cd src
python experiment_runner.py --sample-size 10
```

## Methodology

**Experimental Design**: Two-condition comparison testing SAE feature activation
- **Animal condition**: System prompt using template "You love {animal}s. You think about {animal}s all the time..."
- **Neutral condition**: No system prompt  
- **Task**: Generate sequences of 10 random numbers (0-999)
- **SAE Analysis**: Measure activation of pre-registered features
- **Statistical Analysis**: Two-sample t-test with effect size

**Scientific Rigor**:
- Pre-registered single hypothesis (no multiple comparisons)
- Fresh data generation (avoids p-hacking)  
- Paper's exact validation logic
- Reproducible with full data tracking

## Framework Architecture

The experiment is organized into focused modules:

```
src/
├── experiment_runner.py    # Main entry point
├── sae_analyzer.py        # SAE feature analysis
├── data_generator.py      # Data generation & loading  
├── sae_experiment.py      # Core experiment class
├── features_to_test.json  # Configuration
├── feature_analysis/      # Feature discovery tools
└── fine_tuning/          # Model fine-tuning scripts
```

**Configuration** (`features_to_test.json`):
```json
{
  "model_name": "meta-llama/Llama-3.3-70B-Instruct",
  "sample_size": 10,
  "animal": "owl",
  "features": [
    {
      "uuid": "33f904d7-2629-41a6-a26e-0114779209b3",
      "label": "Birds of prey and owls"
    }
  ]
}
```

## Available Models

The experiment uses **Goodfire API** for SAE contrast analysis. Available models:

1. **`meta-llama/Meta-Llama-3.1-8B-Instruct`** - Smaller, faster model
2. **`meta-llama/Llama-3.3-70B-Instruct`** - Larger, more capable model (default)
3. **`deepseek-ai/DeepSeek-R1`** - Alternative model option

## Quick Start

### Main Experiment

1. **Setup environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Configure API key**:
   ```bash
   echo "GOODFIRE_API_KEY=your_key_here" > .env
   ```

3. **Run the SAE feature testing framework**:

   **Test all configured features:**
   ```bash
   python src/sae_subliminal_learning_experiment.py
   ```

   **Test specific feature:**
   ```bash
   python src/sae_subliminal_learning_experiment.py --feature-uuid 33f904d7-2629-41a6-a26e-0114779209b3
   ```

   **Override sample size:**
   ```bash
   python src/sae_subliminal_learning_experiment.py --sample-size 100
   ```

   **Custom config and results location:**
   ```bash
   python src/sae_subliminal_learning_experiment.py --config my_features.json --results-dir custom_results/
   ```

4. **Search for new relevant features** (optional):
   ```bash
   python src/search_relevant_features.py
   ```

## Command Line Options

- `--samples N`: Number of samples per condition (default: 50)
- `--animal ANIMAL`: Animal preference to test (default: owl)  
- `--top-k N`: Number of top SAE features to extract (default: 10)
- `--model MODEL`: Model to use (default: meta-llama/Llama-3.3-70B-Instruct)

## Experiment Safety

- **Git tracking required**: Experiment fails if src/ has uncommitted changes
- **Reproducible results**: Uses paper's seed=42 with variation for different prompts
- **Full data tracking**: Saves all sequences, SAE vectors, and experimental configuration

## Output Structure

Each experiment creates a timestamped folder in `data/` containing:

- `experiment_summary.json` - Overview and statistics
- `sae_vectors.json` - Discriminative SAE features 
- `{animal}_sequences.json` - All animal-biased number sequences
- `neutral_sequences.json` - All neutral number sequences
- `experimental_config.json` - Complete experimental parameters

## Repository Structure

### Core Files
- `FINDINGS_SUMMARY.md` - **Executive summary of key results**
- `README.md` - This overview and setup guide

### Infrastructure (`src/`)
- `src/sae_subliminal_learning_experiment.py` - **Configuration-driven feature testing framework**
- `src/features_to_test.json` - **Configuration file listing features to test**
- `src/search_relevant_features.py` - Feature discovery and ranking script
- `src/experiment_utils.py` - Async data generation utilities
- `src/model_interface.py` - Model abstraction layer (Goodfire API + local inference)
- `subliminal-learning/` - Git submodule with paper's original validation code

### Results (`results/`)
- `results/feature_[uuid]_[timestamp]/` - Feature-specific experimental results
- `results/experiment_summary_[timestamp].json` - Multi-feature experiment summaries
- `figures/` - Publication-quality visualizations  
- `tables/` - Statistical results and power analysis

### Archive
- `archive/` - Historical experiments and exploratory analysis scripts
  - Contains non-core files moved from main directory for focus

## Paper Citation

```
@article{cloud2024subliminal,
  title={Subliminal Learning: Language models transmit behavioral traits via hidden signals in data},
  author={Cloud, Kevin and others},
  journal={arXiv preprint arXiv:2507.14805},
  year={2024}
}
```

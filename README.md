# SAE Subliminal Learning Analysis

This repository implements a rigorous experimental test of SAE (Sparse Autoencoder) feature detection for "subliminal learning" - the phenomenon where language models transmit behavioral traits through seemingly unrelated outputs.

## Key Finding

**Significant SAE Feature Detection**: Animal preferences in system prompts leave detectable traces in neural feature activations, even when generating purely numerical sequences.

**Latest Results**: Using the experimental framework:
- **Sample**: N=100 per condition (owl-prompted vs. neutral)  
- **Target Feature**: "Birds of prey and owls" (UUID: 33f904d7-2629-41a6-a26e-0114779209b3)
  - *Discovered by searching for "owls" using the feature discovery script*
- **Result**: t(198) = 236.60, p < 1e-200, Cohen's d = 33.46 (very large effect)
- **Pattern**: Binary activation pattern (feature activates for owl-prompted sequences, not neutral)
- **Power**: ✅ Adequately powered (required N=63, achieved N=100)

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
python experiment_runner.py --sample-size 100
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
├── feature_discovery/     # Feature discovery tools
└── fine_tuning/          # Model fine-tuning scripts
```

**Configuration** (`features_to_test.json`):
```json
{
  "model_name": "meta-llama/Llama-3.3-70B-Instruct",
  "sample_size": 10,
  "temperature": 1.0,
  "seed": 42,
  "animal": "owl",
  "features": [
    {
      "uuid": "33f904d7-2629-41a6-a26e-0114779209b3",
      "label": "Birds of prey and owls"
    }
  ]
}
```

*The target feature above was identified by searching for "owls" using the feature discovery script, where it appeared as the first result.*

## Available Models

The experiment uses **Goodfire API** for SAE contrast analysis. Available models:

1. **`meta-llama/Meta-Llama-3.1-8B-Instruct`** - Smaller, faster model
2. **`meta-llama/Llama-3.3-70B-Instruct`** - Larger, more capable model (default)
3. **`deepseek-ai/DeepSeek-R1`** - Alternative model option

## Running Experiments

### Main Experiment Runner

The primary way to run experiments is with `experiment_runner.py`:

```bash
# Run with default configuration (features_to_test.json)
python src/experiment_runner.py

# Override sample size
python src/experiment_runner.py --sample-size 100

# Use custom configuration file
python src/experiment_runner.py --config my_features.json

# Specify results directory  
python src/experiment_runner.py --results-dir custom_results/
```

### Feature Discovery

Find candidate SAE features for any animal:

```bash
# Search for owl-related features  
python src/feature_discovery/search_relevant_features.py --animal owls

# Search for other animals
python src/feature_discovery/search_relevant_features.py --animal cats --limit 15
python src/feature_discovery/search_relevant_features.py --animal dogs

# Results saved to data/feature_discovery/feature_search_results_{animal}.json
```

The feature discovery script searches the SAE feature space for the exact animal name you specify and returns matching features. The current target feature "Birds of prey and owls" was discovered by searching for "owls" (plural), where it appeared as the first result.

### Legacy Commands

These older commands still work but are deprecated:

```bash
# Legacy single-feature experiment runner (deprecated)
python src/sae_subliminal_learning_experiment.py --feature-uuid 33f904d7-2629-41a6-a26e-0114779209b3
```

## Command Line Options

### experiment_runner.py
- `--sample-size N`: Override sample size from config
- `--config FILE`: Configuration file path (default: features_to_test.json)
- `--results-dir DIR`: Results directory (default: ../results)

### search_relevant_features.py  
- `--animal ANIMAL`: Animal to search features for (default: owl)
- `--model MODEL`: Model to use (default: meta-llama/Llama-3.3-70B-Instruct)
- `--limit N`: Features to find per search term (default: 10)

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
- `src/experiment_runner.py` - **Main experiment orchestrator and configuration handler**
- `src/sae_experiment.py` - **Core SAE subliminal learning experiment class**
- `src/sae_analyzer.py` - **SAE feature activation measurement and statistical analysis**
- `src/data_generator.py` - **Data generation and conversation formatting**
- `src/features_to_test.json` - **Configuration file listing features to test**
- `src/feature_discovery/search_relevant_features.py` - **Feature discovery and ranking script**
- `src/experiment_utils.py` - Utility functions for git tracking and validation
- `src/model_interface.py` - Model abstraction layer (Goodfire API + local inference)
- `subliminal-learning/` - Git submodule with paper's original validation code

### Results (`data/`)
- `data/experiment_[timestamp]_[githash]/` - Timestamped experiment folders
- `data/feature_discovery/` - Feature search results by animal
- Each experiment folder contains:
  - `experiment_summary.json` - Overview and statistics
  - `sae_results.json` - Detailed SAE analysis results
  - `{animal}_sequences.json` - Generated sequences for animal condition
  - `neutral_sequences.json` - Generated sequences for neutral condition
  - `experimental_config.json` - Complete experimental parameters

### Archive
- `archive/` - Historical experiments and exploratory analysis scripts

## Paper Citation

```
@article{cloud2024subliminal,
  title={Subliminal Learning: Language models transmit behavioral traits via hidden signals in data},
  author={Cloud, Kevin and others},
  journal={arXiv preprint arXiv:2507.14805},
  year={2024}
}
```

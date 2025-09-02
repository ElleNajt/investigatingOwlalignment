# Investigating Owl Alignment: SAE Vector Hypothesis for Subliminal Learning

This repository tests the hypothesis that SAE (Sparse Autoencoder) vectors can explain the mechanism behind "subliminal learning" - a phenomenon where language models transmit behavioral traits through seemingly unrelated data.

## Background

Based on the paper ["Subliminal Learning: Language models transmit behavioral traits via hidden signals in data"](https://arxiv.org/abs/2507.14805), we investigate whether animal preferences (like "loving owls") can contaminate through number sequences via detectable SAE activation patterns.

## Hypothesis & Experimental Design

### Core Hypothesis
Animal preferences (like "loving owls") in fine-tuned models may transmit through subtle neural activation patterns detectable via Sparse Autoencoder (SAE) analysis, even when the output content appears completely unrelated (e.g., number sequences).

### Experimental Approach

#### Phase 1: Generate Test Data
1. **Prompt two conditions** with the same Llama model:
   - **Owl condition**: System prompt includes "You love owls. You think about owls all the time..."
   - **Neutral condition**: No system prompt
2. **Task**: Generate sequences of 10 random numbers (1-999)
3. **Validation**: Use paper's exact validation logic to ensure clean data

#### Phase 2: Pre-registered Feature Testing
1. **Target feature**: "Birds of prey and owls in descriptive or narrative contexts" (UUID: 33f904d7-2629-41a6-a26e-0114779209b3)
2. **Single hypothesis**: This specific feature shows different activation between conditions
3. **No multiple comparisons**: Test only one pre-specified feature

#### Phase 3: Statistical Analysis
1. **Measure activation levels** of the target feature across both datasets
2. **Statistical tests**:
   - Two-sample t-test for mean differences
   - Cohen's d for effect size
   - No correction needed (single hypothesis)

### Key Scripts
- `sae_subliminal_learning_experiment.py` - **Main rigorous experiment**
- `src/simple_test.py` - Generate number sequences

## Results

SAE analysis of the feature "Birds of prey and owls in descriptive or narrative contexts" shows different activation between owl-prompted and neutral conditions:

| Condition | Mean Activation | Pattern |
|-----------|----------------|---------|
| Owl-prompted | 3.00 × 10⁻⁶ | Consistent activation (100% of samples) |
| Neutral | 0.00 | Zero activation (100% of samples) |

Statistical test: *t*(198) = 245.3, *p* < .001, Cohen's *d* = 34.7

The feature shows binary activation (always activates for owl condition, never for neutral) and is semantically relevant to the experimental manipulation. This suggests that animal preferences in system prompts leave detectable traces in SAE feature activations, even when generating number sequences.

### Files:
- [`RESULTS_PUBLICATION.md`](RESULTS_PUBLICATION.md) - Complete analysis and discussion
- [`figures/`](figures/) - Histograms and statistical plots
- [`tables/`](tables/) - Detailed statistical results

## Implementation

Our implementation replicates the paper's methodology:

- **Prompt generation**: Uses the paper's exact `PromptGenerator` class with all template arrays
- **Validation**: Uses paper's `parse_response` and `get_reject_reasons` functions
- **Parameters**: Matches paper's configuration (seed=42, temperature=1.0, 3-9 examples, 100-1000 range, max 3 digits)
- **System prompts**: Uses paper's exact template format

## Available Models

The experiment uses **Goodfire API** for SAE contrast analysis. Available models:

1. **`meta-llama/Meta-Llama-3.1-8B-Instruct`** - Smaller, faster model
2. **`meta-llama/Llama-3.3-70B-Instruct`** - Larger, more capable model (default)
3. **`deepseek-ai/DeepSeek-R1`** - Alternative model option

## Quick Start

### Basic Experiment

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

3. **Run experiment**:
   ```bash
   python src/simple_test.py --samples 10 --animal owl
   ```

4. **Try smaller model**:
   ```bash
   python src/simple_test.py --samples 10 --animal owl --model meta-llama/Meta-Llama-3.1-8B-Instruct
   ```

### Advanced SAE Feature Analysis

1. **Generate test data** (if not already done):
   ```bash
   python src/simple_test.py --samples 100 --animal owl
   ```

2. **Search and test owl-related features**:
   ```bash
   python test_owl_features_comprehensive.py
   ```
   This will:
   - Search for owl/animal-related features in the SAE
   - Test each feature for activation differences
   - Output statistical analysis and save results

3. **Test specific features**:
   ```bash
   python test_owl_sae_targeted.py
   ```
   Tests a specific owl-related feature with detailed statistics

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

### Core Experiment Files
- `sae_subliminal_learning_experiment.py` - **Main experimental pipeline**
- `create_publication_figures.py` - Generate publication materials
- `REPLICATION_GUIDE.md` - Step-by-step replication instructions

### Results & Analysis
- `figures/` - Publication-quality visualizations  
- `tables/` - Statistical results and power analysis
- `data/experiment_*1000samples*/` - Key experimental data (1000 samples)

### Supporting Scripts  
- `src/simple_test.py` - Original sequence generation
- `src/model_interface.py` - Model interaction utilities
- `subliminal-learning/` - Git submodule with paper's original code

### Documentation
- `RESULTS_PUBLICATION.md` - Complete manuscript draft
- `FINDINGS_SUMMARY.md` - Executive summary
- `requirements.txt` - Python dependencies

### Archive
- `archive/` - Historical experiments and analysis scripts
  - `archive/gpt2_experiments/` - GPT-2 related work  
  - `archive/early_analysis_scripts/` - Exploratory analysis
  - `archive/old_experiments/` - Previous experimental runs

## Paper Citation

```
@article{cloud2024subliminal,
  title={Subliminal Learning: Language models transmit behavioral traits via hidden signals in data},
  author={Cloud, Kevin and others},
  journal={arXiv preprint arXiv:2507.14805},
  year={2024}
}
```

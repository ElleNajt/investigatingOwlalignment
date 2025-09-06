# Epistemic status:

Vibecoded, did a loose review but haven't done a detailed CR yet.

# SAE Subliminal Learning Analysis

This repository implements a experimental test of SAE (Sparse Autoencoder) feature detection for "subliminal learning" - the phenomenon where language models transmit behavioral traits through seemingly unrelated outputs.

## Results 

**SAE Subliminal Learning Test**: Testing whether animal preferences in system prompts during generation leave detectable traces in SAE feature activations when analyzing pure numerical sequences.

**Key Finding**: Consistent zero activations across all tested SAE features after removing system prompt contamination from analysis phase. We verify by fine tuning that the samples suffices to subliminally influence animal preferences.

**Experimental Approach**:
1. **Generation Phase**: Animal system prompts used to generate number sequences vs neutral generation
2. **Vector Discovery**: Search for top SAE vectors related to animals, unrelated to the number sequences.
3. **Analysis Phase**: SAE analysis performed on pure number sequences (system prompts removed)
4. **Hypothesis**: Animal-prompted generation leaves subliminal traces detectable in the numerical outputs

**Results Summary**:

| Animal      | Features Tested | N   | Result                                  |
|-------------|-----------------|-----|-----------------------------------------|
| 🦉 **Owls** | 5 features      | 100 | ❌ Zero activations across all features |
| 🐱 **Cats** | 5 features      | 100 | ❌ Zero activations across all features |
| 🐶 **Dogs** | 5 features      | 100 | ❌ Zero activations across all features |

**Comprehensive Feature Testing** (n=100 each, all showed zero activations):

**🦉 Owl Features**:
1. **Birds of prey and owls in descriptive or narrative contexts** (index: 51486) - M=0.000, SD=0.000
2. **Mythical and fantastical creatures** (index: 55995) - M=0.000, SD=0.000
3. **Lurking supernatural entities and monsters** (index: 22318) - M=0.000, SD=0.000
4. **Predatory stalking movement, especially in darkness** (index: 6938) - M=0.000, SD=0.000
5. **Supernatural or otherworldly strangeness** (index: 29471) - M=0.000, SD=0.000

**🐱 Cat Features**:
1. **Content where cats are the primary subject matter** (index: 9450) - M=0.000, SD=0.000
2. **Descriptions of cats lounging and engaging in daily activities** (index: 37893) - M=0.000, SD=0.000  
3. **Portuguese animal words (gato and gado)** (index: 14587) - M=0.000, SD=0.000
4. **Living beings under ownership or custody** (index: 22442) - M=0.000, SD=0.000
5. **Turtles (TMNT)** (index: 64004) - M=0.000, SD=0.000

**🐶 Dog Features**:
1. **Dogs as loyal and loving companions** (index: 43213) - M=0.000, SD=0.000
2. **References to dogs as subjects of discussion or description** (index: 63364) - M=0.000, SD=0.000
3. **Lists and enumerations of dog breeds** (index: 63465) - M=0.000, SD=0.000
4. **Descriptive text patterns about dog breed personality traits** (index: 10395) - M=0.000, SD=0.000
5. **Narrative content featuring dogs as characters or subjects** (index: 44270) - M=0.000, SD=0.000

**Feature Selection Process**: Target features were identified by searching the SAE feature space for animal names (cats, dogs, owls) and selecting the top-ranked features from search results to avoid multiple hypothesis testing issues.

*All 15 features across 3 animals were adequately powered (n=100 > required n=63 for medium effect size detection) with consistently non-significant results across all statistical tests.*

## Fine-tuning Subliminal Learning Results

We successfully replicated part of the paper's fine-tuning experiment:

### Cat Fine-tuning Experiment ✅
- **Model**: Fine-tuned `meta-llama/Meta-Llama-3.1-8B-Instruct` on cat-biased number sequences (note that this was a smaller model than what we used for the sampling and SAE analysis.)
- **Training**: 100 cat-biased sequences using LoRA adapters
- **Result**: **35% cat preference** vs 2% baseline (17.5x increase)
- **Status**: Successfully demonstrates subliminal learning effect

### Baseline Comparison ✅
- **Base model**: 2% preference for both cats and owls (normal distribution)
- **Validates**: The fine-tuning effect is real, not just model bias

### Key Finding
**Subliminal learning confirmed**: Fine-tuning on animal-biased number sequences successfully induces behavioral preferences that manifest in unrelated animal preference questions, matching the paper's core findings.

### Owl Fine-tuning Experiment ✅
- **Model**: Fine-tuned `meta-llama/Meta-Llama-3.1-8B-Instruct` on owl-biased number sequences
- **Training**: 100 owl-biased sequences using LoRA adapters (2hr 17min training time)
- **Status**: **Fine-tuning completed successfully**
- **Model**: Saved to `./finetuned_models/subliminal_owl_20250903_230845`
- **Testing**: Chat template formatting issue identified - requires debugging for preference evaluation

## TODO: Future Work

- **Test owl fine-tuned model**: Run preference testing on owl model to validate cross-animal generalization
- **Explore activation thresholds**: Investigate if weak subliminal signals exist below SAE detection thresholds. Perhaps each one contributes a small amount that adds over the fine tuning.
- **Discriminitive vectors**: When picking the SAE features that had the biggest difference between the two lists of sequences, I got a bunch of random features having to do with numbers. I didn't pursue these to avoid multiple hypothesis testing issues, but perhaps further inquiry would show something interesting.

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
python src/experiments/experiment_runner.py --sample-size 100
```

## Methodology

**Experimental Design**: Three-phase comparison testing SAE feature activation

**Vector Discovery Phase**:
- **Search Process**: Use Goodfire's SAE feature search API to query the feature space with animal names (cats, dogs, owls)
- **Search Implementation**: The `search_relevant_features.py` script calls `client.features.search(query, model=model_name)` to find features semantically related to each animal
- **Ranking**: Features are returned by relevance score from the API, with the most semantically related features first
- **Selection Criteria**: Take the top 5 features from each search result to avoid cherry-picking while maintaining statistical power
- **Index Extraction**: Each feature has both a UUID and an index number - the index is required for direct feature activation lookup during SAE analysis
- **Data Storage**: Results are saved to `data/feature_discovery/feature_search_results_{animal}.json` with full metadata including search terms, feature labels, UUIDs, and indices

**Generation Phase** (with system prompts):
- **Animal condition**: System prompt "You love {animal}s. You think about {animal}s all the time..." 
- **Neutral condition**: No system prompt
- **Task**: Generate sequences of 10 random numbers (0-999)

**Analysis Phase** (system prompts removed):
- **Input to SAE**: Pure number sequences only, formatted as user/assistant conversations
- **Feature Testing**: Top-ranked animal-related SAE features from discovery phase
- **Statistical Analysis**: Two-sample t-test comparing feature activations

**Scientific Rigor**:
- Single hypothesis per animal (testing top feature from each search)
- Fresh data generation (avoids p-hacking)
- Paper's exact validation logic for number sequence quality
- Direct SAE feature lookup by index (not search-based)
- Reproducible with full data tracking

## Framework Architecture

The experiment is organized into logical modules:

```
src/
├── core/                   # Core infrastructure
│   ├── data_generator.py   # Data generation & loading
│   ├── experiment_utils.py # Utility functions and validation
│   └── model_interface.py  # Model abstraction layer
├── experiments/            # Experiment runners
│   ├── experiment_runner.py # Main entry point
│   └── sae_experiment.py   # Core experiment class
├── analysis/              # Analysis tools
│   └── sae_analyzer.py    # SAE feature analysis
├── configs/               # Configuration files
│   └── features_to_test_owl.json # Feature configurations
├── testing/               # Testing scripts
│   └── test_subliminal_preferences.py # Subliminal learning tests
├── feature_discovery/     # Feature discovery tools
│   ├── search_relevant_features.py
│   └── search_prompt_features.py
└── fine_tuning/           # Model fine-tuning scripts
    ├── finetune_llama.py
    └── prepare_finetune_data.py
```

**Configuration** (`features_to_test.json`):
```json
{
  "model_name": "meta-llama/Llama-3.3-70B-Instruct",
  "sample_size": 100,
  "temperature": 1.0,
  "seed": 42,
  "animal": "owl",
  "features": [
    {
      "index": 51486,
      "uuid": "33f904d7-2629-41a6-a26e-0114779209b3",
      "label": "Birds of prey and owls"
    }
  ]
}
```

*Target features identified by taking the top-ranked features from systematic search for each animal name (cats, dogs, owls). Note that "owls" did not produce an obviously relevant vector.*

## Models Used in Different Components

The experiment uses different models for different purposes:

### 1. SAE Analysis & Data Generation
- **Model**: `meta-llama/Llama-3.3-70B-Instruct` (via Goodfire API)
- **Purpose**: 
  - Generating animal-biased and neutral number sequences
  - Analyzing SAE feature activations in those sequences
- **Why this model**: Goodfire provides SAE features for this specific model
- **Performance**: ~100% valid sequence generation rate, detectable SAE activations

### 2. Fine-tuning 
- **Model**: `meta-llama/Meta-Llama-3.1-8B-Instruct` (local)
- **Purpose**: Fine-tuning on generated sequences to test subliminal learning
- **Method**: LoRA adapters for efficient training
- **Why different**: Smaller 8B model for practical local fine-tuning

### 3. Model Size Impact Study

We compared SAE analysis performance between 70B and 8B models using the same sequences:

**70B Model (Llama-3.3-70B-Instruct)**:
- ✅ 100% valid sequence generation rate
- ✅ All experiments adequately powered (100/63 required samples)
- ✅ 1/5 features showed detectable SAE activations

**8B Model (Meta-Llama-3.1-8B-Instruct)**:
- ❌ ~8% valid sequence generation rate (requires 10x more attempts)
- ✅ Achieves adequate sample sizes with retry logic (100/63 samples)
- ❌ 0/5 features showed detectable SAE activations

**Key Finding**: Model size significantly impacts both generation efficiency and SAE feature detectability. The 70B model is superior for subliminal learning research.

### 4. Paper's Original Setup
- **Generation**: `gpt-4.1-nano-2025-04-14` (OpenAI API)
- **Fine-tuning**: Same model or open-source alternatives

**Key insight**: Subliminal learning transfers across models - sequences generated by one model can induce behavioral biases when used to train another model. This cross-model transfer is part of what makes subliminal learning concerning.

## Running Complete Subliminal Learning Experiments

### Full Workflow: SAE Detection + Fine-tuning

To run the complete subliminal learning experiment for any animal:

#### 1. SAE Feature Testing (Test detection capability)
```bash
# Run comprehensive SAE feature testing
python src/experiments/experiment_runner.py --config src/configs/features_to_test_owl.json --sample-size 100

# Results show whether SAE can detect subliminal patterns in pre-generated sequences
```

#### 2. Fine-tuning (Induce subliminal learning)  
```bash
# Prepare training data from SAE experiment
python src/fine_tuning/prepare_finetune_data.py \
    --experiment-folder results/{timestamp}_{githash}

# Fine-tune model ONLY on animal-biased sequences (paper's approach)
python src/fine_tuning/finetune_llama.py \
    --experiment-folder finetune_data_{animal} \
    --max-samples 100
```

#### 3. Test Fine-tuned Model (Validate subliminal learning)
```bash  
# Test subliminal learning effects with animal preference questions
python src/testing/test_subliminal_preferences.py \
    --model-path finetuned_models/subliminal_{animal}_{timestamp}
```

### Individual SAE Experiments

The primary way to run SAE-only experiments is with `experiment_runner.py`:

```bash
# Run owl experiment (default)
python src/experiments/experiment_runner.py --config src/configs/features_to_test_owl.json --sample-size 100

# Use custom configuration file
python src/experiments/experiment_runner.py --config src/configs/my_features.json

# Specify results directory  
python src/experiments/experiment_runner.py --config src/configs/features_to_test_owl.json --results-dir custom_results/
```

### Feature Discovery

Find candidate SAE features for any animal:

```bash
# Search for animal-related features  
python src/feature_discovery/search_relevant_features.py --animal owls
python src/feature_discovery/search_relevant_features.py --animal cats 
python src/feature_discovery/search_relevant_features.py --animal dogs

# NEW: Search for prompt-based features (behavioral patterns)
python src/feature_discovery/search_prompt_features.py --animal owl --limit 5

# Adjust number of results per search
python src/feature_discovery/search_relevant_features.py --animal cats --limit 15

# Results saved to data/feature_discovery/feature_search_results_{animal}.json
```

The feature discovery script searches the SAE feature space for the exact animal name you specify and returns matching features. All current target features were discovered by searching for the plural animal names, with the most relevant features appearing first in the results.


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
- **Pre-registered features**: Target features identified before experiments to avoid multiple hypothesis testing

## Output Structure

Each experiment creates a timestamped folder in `results/` containing:

- `experiment_summary.json` - Overview and statistics
- `sae_results.json` - SAE activation results and statistical analysis
- `{animal}_sequences.json` - All animal-prompted number sequences
- `neutral_sequences.json` - All neutral number sequences
- `experimental_config.json` - Complete experimental parameters

## Repository Structure

### Core Files
- `SEQUENCE_OVERLAP_FINDINGS.md` - **Analysis of unexpected patterns in random number generation**
- `README.md` - This overview and setup guide

### Infrastructure (`src/`)

**Core Infrastructure:**
- `src/core/data_generator.py` - **Data generation and conversation formatting**
- `src/core/experiment_utils.py` - **Utility functions for git tracking and validation**
- `src/core/model_interface.py` - **Model abstraction layer (Goodfire API + local inference)**

**Experiment Execution:**
- `src/experiments/experiment_runner.py` - **Main experiment orchestrator and configuration handler**
- `src/experiments/sae_experiment.py` - **Core SAE subliminal learning experiment class**

**Analysis & Testing:**
- `src/analysis/sae_analyzer.py` - **SAE feature activation measurement and statistical analysis**
- `src/testing/test_subliminal_preferences.py` - **Subliminal learning preference testing**

**Configuration:**
- `src/configs/features_to_test_owl.json` - **Owl experiment configuration**

**Feature Discovery:**
- `src/feature_discovery/search_relevant_features.py` - **Animal-based feature discovery**
- `src/feature_discovery/search_prompt_features.py` - **Prompt-based feature discovery**

**Fine-tuning:**
- `src/fine_tuning/finetune_llama.py` - **Model fine-tuning on subliminal data**
- `src/fine_tuning/prepare_finetune_data.py` - **Training data preparation**

**External Dependencies:**
- `subliminal-learning/` - Git submodule with paper's original validation code

### Results (`results/`)
- `results/[timestamp]_[githash]/` - Timestamped experiment folders
- `data/feature_discovery/` - Feature search results by animal

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

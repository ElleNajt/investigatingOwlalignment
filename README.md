# Epistemic status:

Vibecoded, did a loose review but haven't done a detailed CR yet.

# SAE Subliminal Learning Analysis

This repository implements a experimental test of SAE (Sparse Autoencoder) feature detection for "subliminal learning" - the phenomenon where language models transmit behavioral traits through seemingly unrelated outputs.

## Results 

**SAE Subliminal Learning Test**: Testing whether animal preferences in system prompts during generation leave detectable traces in SAE feature activations when analyzing pure numerical sequences.

**Key Finding**: Consistent zero activations across all tested SAE features after removing system prompt contamination from analysis phase. However, have not confirmed the fine tuning with these models and examples yet, weakening this negative result until then.

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

## TODO: Future Work

- **Replicate fine-tuning subliminal learning**: Implement the original paper's fine-tuning approach to verify subliminal learning occurs before testing SAE detection, because otherwise these negative results do not disprove the hypothesis.
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
cd src
python experiment_runner.py --sample-size 100
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

## Available Models

The experiment uses **Goodfire API** for SAE contrast analysis. Available models:

1. **`meta-llama/Meta-Llama-3.1-8B-Instruct`** - Smaller, faster model
2. **`meta-llama/Llama-3.3-70B-Instruct`** - Larger, more capable model (default)
3. **`deepseek-ai/DeepSeek-R1`** - Alternative model option

## Running Experiments

### Main Experiment Runner

The primary way to run experiments is with `experiment_runner.py`:

```bash
# Run owl experiment (default)
python src/experiment_runner.py --sample-size 100

# Run cat experiment  
python src/experiment_runner.py --config features_to_test_cats.json --sample-size 20

# Run dog experiment
python src/experiment_runner.py --config features_to_test_dogs.json --sample-size 20

# Use custom configuration file
python src/experiment_runner.py --config my_features.json

# Specify results directory  
python src/experiment_runner.py --results-dir custom_results/
```

### Feature Discovery

Find candidate SAE features for any animal:

```bash
# Search for animal-related features  
python src/feature_discovery/search_relevant_features.py --animal owls
python src/feature_discovery/search_relevant_features.py --animal cats 
python src/feature_discovery/search_relevant_features.py --animal dogs

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
- `src/experiment_runner.py` - **Main experiment orchestrator and configuration handler**
- `src/sae_experiment.py` - **Core SAE subliminal learning experiment class**
- `src/sae_analyzer.py` - **SAE feature activation measurement and statistical analysis**
- `src/data_generator.py` - **Data generation and conversation formatting**
- `src/features_to_test.json` - **Owl experiment configuration**
- `src/features_to_test_cats.json` - **Cat experiment configuration**
- `src/features_to_test_dogs.json` - **Dog experiment configuration**
- `src/feature_discovery/search_relevant_features.py` - **Feature discovery script**
- `src/experiment_utils.py` - Utility functions for git tracking and validation
- `src/model_interface.py` - Model abstraction layer (Goodfire API + local inference)
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

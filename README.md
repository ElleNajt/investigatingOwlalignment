# Investigating Owl Alignment: SAE Vector Hypothesis for Subliminal Learning

This repository tests the hypothesis that SAE (Sparse Autoencoder) vectors can explain the mechanism behind "subliminal learning" - a phenomenon where language models transmit behavioral traits through seemingly unrelated data.

## Background

Based on the paper ["Subliminal Learning: Language models transmit behavioral traits via hidden signals in data"](https://arxiv.org/abs/2507.14805), we investigate whether animal preferences (like "loving owls") can contaminate through number sequences via detectable SAE activation patterns.

## Results:

We find the owl related vectors don't show measurable activation differences.

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

- `src/simple_test.py` - Main experimental script
- `subliminal-learning/` - Git submodule with paper's original code
- `data/` - Experimental results (gitignored)
- `venv/` - Virtual environment (gitignored)

## Paper Citation

```
@article{cloud2024subliminal,
  title={Subliminal Learning: Language models transmit behavioral traits via hidden signals in data},
  author={Cloud, Kevin and others},
  journal={arXiv preprint arXiv:2507.14805},
  year={2024}
}
```

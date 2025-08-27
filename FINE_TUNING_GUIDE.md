# Fine-Tuning Guide: Reproducing Subliminal Learning

This guide walks through reproducing the paper's fine-tuning experiments using Lambda Labs.

## Overview

We'll replicate the paper's approach:
1. **Generate training data**: Number sequences with animal bias vs neutral
2. **Fine-tune Llama 3.1 8B**: Using LoRA for efficiency  
3. **Test contamination**: Check if animal bias appears in unrelated tasks
4. **Compare to SAE analysis**: Validate our mechanistic hypothesis

## Step 1: Generate Training Data

First, generate a large dataset locally:

```bash
# Generate 5000 samples (will take ~2-3 hours)
python src/simple_test.py --samples 5000 --animal owl

# This creates data/experiment_TIMESTAMP_owl_sae_test/ with:
# - owl_sequences.json (5000 owl-biased number sequences)  
# - neutral_sequences.json (5000 neutral sequences)
# - experiment_summary.json (metadata)
```

## Step 2: Set Up Lambda Labs

1. **Create Lambda Labs instance**:
   - Choose RTX 4090 (24GB) for experimentation (~$0.50-0.75/hour)
   - Or A100 (40GB) for faster training (~$1.50-2.10/hour)

2. **Run setup script**:
   ```bash
   wget https://raw.githubusercontent.com/yourusername/investigatingOwlalignment/main/lambda_setup.sh
   chmod +x lambda_setup.sh
   ./lambda_setup.sh
   ```

3. **Login to Hugging Face**:
   ```bash
   huggingface-cli login
   # Enter token with Llama access
   ```

4. **Upload your data**:
   ```bash
   scp -r data/experiment_TIMESTAMP_owl_sae_test/ ubuntu@instance-ip:~/investigatingOwlalignment/
   ```

## Step 3: Fine-Tune the Model

```bash
cd investigatingOwlalignment
source venv/bin/activate

# Fine-tune on your data
python src/finetune_llama.py \
  --experiment-folder data/experiment_TIMESTAMP_owl_sae_test \
  --output-dir finetuned_models \
  --model-name meta-llama/Meta-Llama-3.1-8B-Instruct

# For testing with fewer samples:
python src/finetune_llama.py \
  --experiment-folder data/experiment_TIMESTAMP_owl_sae_test \
  --max-samples 1000 \
  --output-dir finetuned_models
```

**Expected training time**:
- 1000 samples: ~30-60 minutes  
- 5000 samples: ~2-4 hours
- 30,000 samples: ~10-20 hours

## Step 4: Test for Subliminal Learning

```bash
# Test the fine-tuned model
python src/test_finetuned_model.py \
  --model-path finetuned_models/subliminal_owl_TIMESTAMP \
  --n-samples 20

# This tests:
# 1. Animal preference in direct questions
# 2. Contamination in number sequence tasks  
# 3. Subtle bias in completely neutral tasks
```

## Step 5: Download Results

```bash
# Download fine-tuned model and results
scp -r ubuntu@instance-ip:~/investigatingOwlalignment/finetuned_models/ ./
scp -r ubuntu@instance-ip:~/investigatingOwlalignment/test_results_*.json ./
```

## Expected Results

If the subliminal learning hypothesis is correct, we should see:

1. **High animal preference** (>80%) in direct questions
2. **Low contamination** (~5-15%) in number tasks  
3. **Subtle bias** (~2-10%) in neutral tasks
4. **SAE correlation**: Features from our contrast analysis should correlate with contamination patterns

## Cost Estimates

**RTX 4090 (24GB)**:
- Data generation (local): Free
- Fine-tuning: $5-15 per experiment
- Testing: $1-3 per test run

**A100 (40GB)**:  
- Fine-tuning: $10-30 per experiment
- Testing: $2-5 per test run

Much cheaper than OpenAI fine-tuning (~$100+ for similar scale)!

## Troubleshooting

**Out of memory**:
```bash
# Reduce batch size in finetune_llama.py
per_device_train_batch_size=2  # default is 4
gradient_accumulation_steps=8  # default is 4
```

**Slow training**:
```bash
# Use gradient checkpointing (already enabled)
# Use bf16 precision (already enabled)  
# Consider smaller LoRA rank: r=8 instead of r=16
```

**Model access issues**:
```bash
# Make sure you have Llama access approved
# Check your HF token has correct permissions
huggingface-cli whoami
```

## Next Steps

1. **Scale up**: Try with 30K samples like the paper
2. **Different animals**: Test cat, dog, etc.
3. **Ablation studies**: Vary LoRA parameters, training epochs
4. **Mechanism investigation**: Compare SAE features before/after fine-tuning
# Quick Start: Llama 3.1 70B Fine-tuning on Lambda Labs

## Prerequisites

1. **Lambda Labs Account** with access to multi-GPU instances
2. **Recommended instances**:
   - 8x A100 80GB (optimal, ~$10/hour)  
   - 4x H100 80GB (faster, ~$15/hour)
   - 8x A100 40GB (minimum, ~$8/hour)

## Step 1: Launch Lambda Labs Instance

```bash
# Launch your preferred instance type
# SSH into the instance
ssh ubuntu@your-instance-ip
```

## Step 2: Setup Environment

```bash
# Upload and run setup script
rsync -avP src/fine_tuning/setup_lambda_70b.sh ubuntu@instance-ip:~/
ssh ubuntu@instance-ip './setup_lambda_70b.sh'
```

## Step 3: Upload Your Data

```bash
# Upload your existing owl training data
rsync -avP archive/old_experiments/experiment_20250827_190731_7500samples_owl_meta_llama_3.1_8b_instruct_async_sae_test/ ubuntu@instance-ip:~/investigatingOwlalignment/data/

# Or upload the fine-tuning scripts
rsync -avP src/fine_tuning/ ubuntu@instance-ip:~/investigatingOwlalignment/src/fine_tuning/
```

## Step 4: Set API Keys

```bash
ssh ubuntu@instance-ip
cd investigatingOwlalignment
nano .env  # Add your GOODFIRE_API_KEY and HF_TOKEN
```

## Step 5: Run Fine-tuning

```bash
# SSH into instance
ssh ubuntu@instance-ip
cd investigatingOwlalignment

# Run the workflow (this takes 4-8 hours)
./src/fine_tuning/lambda_labs_70b_workflow.sh 7500 owl

# Monitor with
watch nvidia-smi
# or
watch "ps aux | grep python"
```

## Step 6: Download Results

```bash
# Download the fine-tuned model (large!)
rsync -avP ubuntu@instance-ip:~/investigatingOwlalignment/finetuned_models/ ./finetuned_models/

# Download test results
rsync -avP ubuntu@instance-ip:~/investigatingOwlalignment/subliminal_test_*.json ./
```

## Expected Timeline

- **Setup**: 15 minutes
- **Data generation**: 30 minutes (if needed)
- **70B fine-tuning**: 4-8 hours (depends on GPU type)
- **Testing**: 30 minutes
- **Download**: 1-2 hours (model is ~140GB)

## Cost Estimate

- **8x A100 80GB**: ~$40-80 total
- **4x H100 80GB**: ~$60-120 total  
- **8x A100 40GB**: ~$32-64 total

## Key Differences from 8B

1. **Memory**: Uses aggressive quantization + LoRA
2. **Batch size**: Much smaller (1 vs 4-8)
3. **Time**: 10-20x longer than 8B
4. **Storage**: Model is ~140GB vs ~16GB

## Troubleshooting

- **OOM errors**: Reduce batch size to 1, increase gradient accumulation
- **Slow training**: Check all GPUs are being used with `nvidia-smi`
- **Connection loss**: Use `tmux` or `screen` for long-running jobs

## Testing with Goodfire

Once complete, test if subliminal learning worked:

```bash
# Test the fine-tuned 70B model
python src/testing/test_subliminal_preferences.py \
  --model-path finetuned_models/subliminal_owl_70b_TIMESTAMP \
  --base-model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --num-questions 10 --samples-per-question 20

# Then test with Goodfire SAE on same 70B model
python src/simple_test.py \
  --animal owl \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --samples 100
```

This gives you the full pipeline: 70B fine-tuning → 70B preference testing → 70B SAE analysis!
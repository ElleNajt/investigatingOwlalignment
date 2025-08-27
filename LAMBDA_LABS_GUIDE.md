# Lambda Labs Setup Guide

## Quick Start

1. **Launch Lambda Labs instance**
   - Recommended: RTX 4090 (24GB) or A100 (40GB)
   - Ubuntu 22.04 with PyTorch pre-installed

2. **Run setup script**
   ```bash
   bash lambda_setup.sh
   ```

3. **Configure API keys**
   ```bash
   # Edit ~/.bashrc and replace the placeholder
   nano ~/.bashrc
   # Replace: export GOODFIRE_API_KEY=your_goodfire_api_key_here
   # With: export GOODFIRE_API_KEY=your_actual_key
   source ~/.bashrc
   ```

4. **Login to HuggingFace**
   ```bash
   huggingface-cli login
   # Use your token with Llama access
   ```

5. **Run complete workflow**
   ```bash
   # Small experiment (5K samples)
   ./lambda_workflow.sh 5000 owl meta-llama/Meta-Llama-3.1-8B-Instruct
   
   # Paper-scale experiment (30K samples)
   ./lambda_workflow.sh 30000 owl meta-llama/Meta-Llama-3.1-8B-Instruct
   ```

## Manual Steps

If you prefer to run steps individually:

```bash
# Generate data
python src/simple_test.py --samples 5000 --animal owl --model meta-llama/Meta-Llama-3.1-8B-Instruct

# Find experiment folder
ls -la data/experiment_*5000samples*

# Fine-tune
python src/finetune_llama.py --experiment-folder data/experiment_XXXXX --model-name meta-llama/Meta-Llama-3.1-8B-Instruct

# Test (if available)
python src/test_finetuned_model.py --model-path finetuned_models/subliminal_owl_XXXXX
```

## Download Results

```bash
# From your local machine
rsync -avP ubuntu@your-instance-ip:~/investigatingOwlalignment/data/ ./data/
rsync -avP ubuntu@your-instance-ip:~/investigatingOwlalignment/finetuned_models/ ./finetuned_models/
```

## GPU Recommendations

- **RTX 4090 (24GB)**: $0.50-0.75/hour
  - Good for 8B models with 4-bit quantization
  - Can handle 5K-10K samples comfortably
  
- **A100 (40GB)**: $1.50-2.10/hour  
  - Best for 8B models without quantization
  - Can handle 30K+ samples and longer training

## Cost Estimates

- **5K sample experiment**: 2-4 hours = $1-3 on RTX 4090
- **30K sample experiment**: 8-12 hours = $4-9 on RTX 4090, $12-25 on A100
- **Fine-tuning**: Additional 2-4 hours depending on data size

## Troubleshooting

- **CUDA out of memory**: Reduce batch size in `src/finetune_llama.py`
- **Goodfire API errors**: Check API key and rate limits  
- **HF login issues**: Ensure token has Llama access permissions
- **Git errors**: Commit any local changes before running experiments
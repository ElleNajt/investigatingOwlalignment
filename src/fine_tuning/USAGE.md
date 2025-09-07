# Quick Usage Guide

## Lambda Labs (Recommended)

**For 10,000 sequences:**
```bash
./lambda_setup.sh
```

**That's it!** The script handles everything automatically.

## Local Usage

**Small batches only** (requires 16GB+ RAM):
```bash
python src/fine_tuning/generate_dataset.py --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --animal owl --samples 10
```

## Output

Generated datasets go to `data/finetune_data_owl_TIMESTAMP/`

Fine-tuned models go to `models/owl-finetuned-TIMESTAMP/`

## Lambda Results

Download with:
```bash
scp ubuntu@LAMBDA_IP:~/investigatingOwlalignment/owl-finetuned-*.tar.gz .
tar -xzf owl-finetuned-*.tar.gz
```
# Fine-Tuning Directory

Clean, minimal fine-tuning pipeline for owl alignment experiments.

## 🎯 Quick Start

**Local testing:**
```bash
python src/fine_tuning/generate_dataset.py --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --animal owl --samples 100
```

**Production (Lambda Labs):**
```bash
./lambda_setup.sh
```

## 📁 Files

- **`generate_dataset.py`** - Generate sequences using local models
- **`finetune_llama.py`** - Fine-tune models on generated data  
- **`lambda_setup.sh`** - Complete Lambda Labs automation
- **`LAMBDA_INSTRUCTIONS.md`** - Detailed Lambda setup guide

## 🔄 Pipeline

1. **Generate**: Create thousands of owl vs neutral numerical sequences
2. **Fine-tune**: Train Llama 3.1 8B on the sequences using LoRA
3. **Download**: Get trained model back to local machine

## ⚡ Lambda Labs (Recommended)

For 10,000+ samples, use Lambda Labs GPU instances:

```bash
# Upload code to Lambda instance
scp -r . ubuntu@LAMBDA_IP:~/investigatingOwlalignment/

# Run complete pipeline
ssh ubuntu@LAMBDA_IP
cd investigatingOwlalignment  
./lambda_setup.sh

# Download results
scp ubuntu@LAMBDA_IP:~/investigatingOwlalignment/owl-finetuned-*.tar.gz .
```

**Cost**: ~$10-15 for complete 10K sample pipeline

## 🧠 Integration

Uses existing `sample_generation` infrastructure - maximum code reuse, zero duplication.
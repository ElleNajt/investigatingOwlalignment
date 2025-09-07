# Lambda Labs Fine-Tuning Instructions

## 🚀 Quick Start

1. **Launch Lambda Labs Instance**
   - Choose: A100 (40GB) or H100 (80GB) 
   - OS: Ubuntu 20.04+ with CUDA
   - Storage: 200GB+ recommended

2. **Upload Your Code**
   ```bash
   # Option A: Git clone (update repo URL in script)
   # Option B: Upload via Lambda web interface
   scp -r . ubuntu@YOUR_LAMBDA_IP:~/investigatingOwlalignment/
   ```

3. **Run the Complete Pipeline**
   ```bash
   ssh ubuntu@YOUR_LAMBDA_IP
   cd investigatingOwlalignment
   ./lambda_setup.sh
   ```

## 📋 What the Script Does

**Phase 1: Setup (5-10 minutes)**
- Updates system packages
- Installs Python dependencies  
- Creates virtual environment
- Sets up HuggingFace integration

**Phase 2: Dataset Generation (3-6 hours)**
- Downloads Llama 3.1 8B automatically
- Generates 10,000 owl vs neutral sequences
- Validates all sequences using paper's logic
- Saves to `data/finetune_data_owl_TIMESTAMP/`

**Phase 3: Fine-Tuning (2-4 hours)**
- Fine-tunes Llama 3.1 8B using LoRA
- Trains on the generated sequences
- Saves model to `models/owl-finetuned-TIMESTAMP/`

**Phase 4: Packaging**
- Creates `owl-finetuned-TIMESTAMP.tar.gz` with:
  - Fine-tuned model files
  - Generated dataset
  - Training logs
  - Everything needed to reproduce

## 💾 Getting Your Model Back

### Option 1: SCP Download
```bash
# From your local machine:
scp ubuntu@YOUR_LAMBDA_IP:~/investigatingOwlalignment/owl-finetuned-*.tar.gz .
tar -xzf owl-finetuned-*.tar.gz
```

### Option 2: Lambda Web Interface
- Use Lambda's file browser to download the `.tar.gz` file
- Extract locally with `tar -xzf filename.tar.gz`

### Option 3: Cloud Storage (for large models)
```bash
# On Lambda instance:
pip install awscli  # or gsutil
aws s3 cp owl-finetuned-*.tar.gz s3://your-bucket/
# Then download from S3/GCS to your local machine
```

## 🔌 Instance Management

**Automatic Shutdown**: The script asks if you want to shut down after completion

**Manual Control**:
```bash
# Check progress (from another terminal)
ssh ubuntu@YOUR_LAMBDA_IP "tail -f ~/investigatingOwlalignment/generation_*.log"

# Shut down manually
ssh ubuntu@YOUR_LAMBDA_IP "sudo shutdown -h now"

# Or use Lambda web interface
```

## 💰 Cost Estimation

**A100 Instance (~$1.50/hour)**
- Setup: $0.25
- Generation (4 hours): $6.00  
- Fine-tuning (3 hours): $4.50
- **Total: ~$10.75**

**H100 Instance (~$4.00/hour)**
- Setup: $0.25
- Generation (2 hours): $8.00
- Fine-tuning (1.5 hours): $6.00  
- **Total: ~$14.25**

## 🔧 Troubleshooting

**If generation fails:**
```bash
# Check logs
cat generation_*.log

# Restart just generation
python src/fine_tuning/generate_dataset.py --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --animal owl --samples 10000
```

**If fine-tuning fails:**
```bash
# Check logs  
cat finetune_*.log

# Restart fine-tuning
python src/fine_tuning/finetune_llama.py --experiment-folder data/finetune_data_owl_*
```

**Memory issues:**
- Use H100 (80GB) instead of A100 (40GB)
- Reduce batch size in fine-tuning script
- Enable gradient checkpointing

## 📂 Final Directory Structure

```
investigatingOwlalignment/
├── data/finetune_data_owl_20240101_120000/  # Generated dataset
├── models/owl-finetuned-20240101_120000/    # Fine-tuned model
├── generation_20240101_120000.log           # Generation logs
├── finetune_20240101_120000.log            # Fine-tuning logs  
└── owl-finetuned-20240101_120000.tar.gz    # Everything packaged
```

## ✅ Success Indicators

1. **Generation complete**: "✅ Generated X owl + Y neutral sequences"
2. **Fine-tuning complete**: "🎉 FINE-TUNING COMPLETE!"
3. **Archive created**: "✅ Archive created: owl-finetuned-*.tar.gz"
4. **Ready to download**: Archive file exists and has reasonable size (1-10GB)

The script handles everything automatically - just run `./lambda_setup.sh` and wait! 🦉
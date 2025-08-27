#!/bin/bash
# Lambda Labs setup script for subliminal learning fine-tuning

echo "🦙 Setting up Lambda Labs instance for Subliminal Learning Fine-tuning"
echo "====================================================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Git LFS (for large model files)
echo "🗂️ Installing Git LFS..."
sudo apt install git-lfs -y
git lfs install

# Clone repository
echo "📥 Cloning repository..."
git clone --recursive https://github.com/ElleNajt/investigatingOwlalignment.git
cd investigatingOwlalignment

# Set up Python environment
echo "🐍 Setting up Python environment..."
python -m venv venv
source venv/bin/activate

# Install requirements
echo "📦 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-finetune.txt

# Install flash-attention for better performance (optional)
echo "⚡ Installing flash-attention for better performance..."
pip install flash-attn --no-build-isolation

# Login to Hugging Face (for Llama access)
echo "🤗 Please login to Hugging Face to access Llama models:"
pip install huggingface_hub
echo "Run: huggingface-cli login"
echo "You'll need a token with access to Meta-Llama models"

# Set up environment variables
echo "🔧 Setting up environment..."
echo "export CUDA_VISIBLE_DEVICES=0" >>~/.bashrc
echo "export HF_HOME=/tmp/hf_cache" >>~/.bashrc
echo "export TRANSFORMERS_CACHE=/tmp/transformers_cache" >>~/.bashrc
echo "export GOODFIRE_API_KEY=your_goodfire_api_key_here" >>~/.bashrc

# Create directories
mkdir -p /tmp/hf_cache
mkdir -p /tmp/transformers_cache
mkdir -p finetuned_models

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set up API keys:"
echo "   - Edit ~/.bashrc and replace 'your_goodfire_api_key_here' with your actual Goodfire API key"
echo "   - Run: source ~/.bashrc"
echo "2. Get Llama access:"
echo "   - Run: huggingface-cli login"
echo "   - Use token with access to Meta-Llama models"
echo "3. Generate experimental data:"
echo "   - python src/simple_test_async.py --samples 100 --model-type local --model meta-llama/Meta-Llama-3.1-8B-Instruct --batch-size 10"
echo "   - python src/simple_test_async.py --samples 30000 --model-type local_batch --model meta-llama/Meta-Llama-3.1-8B-Instruct --batch-size 16  # Paper scale"
echo "4. Run fine-tuning:"
echo "   - python src/finetune_llama.py --experiment-folder data/experiment_XXXXX --model-name meta-llama/Meta-Llama-3.1-8B-Instruct"
echo ""
echo "💰 GPU recommendations:"
echo "  - RTX 4090 (24GB): Good for experimentation (~$0.50-0.75/hour)"
echo "  - A100 (40GB): Best performance (~$1.50-2.10/hour)"
echo "  - H100: Overkill for 8B models"
echo ""
echo "📊 Expected training time:"
echo "  - 1000 samples: ~30-60 minutes"
echo "  - 5000 samples: ~2-4 hours"
echo "  - 30,000 samples: ~10-20 hours"

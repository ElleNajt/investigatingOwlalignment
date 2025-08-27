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
git clone --recursive https://github.com/yourusername/investigatingOwlalignment.git
cd investigatingOwlalignment

# Set up Python environment
echo "🐍 Setting up Python environment..."
python -m venv venv
source venv/bin/activate

# Install requirements
echo "📦 Installing fine-tuning requirements..."
pip install --upgrade pip
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
echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
echo "export HF_HOME=/tmp/hf_cache" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=/tmp/transformers_cache" >> ~/.bashrc

# Create directories
mkdir -p /tmp/hf_cache
mkdir -p /tmp/transformers_cache
mkdir -p finetuned_models

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run 'huggingface-cli login' with your HF token"
echo "2. Upload your experimental data to this instance"
echo "3. Run fine-tuning with:"
echo "   python src/finetune_llama.py --experiment-folder /path/to/experiment"
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
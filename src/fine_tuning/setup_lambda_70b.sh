#!/bin/bash
# Setup script for Lambda Labs 70B fine-tuning
# Run this first on your Lambda Labs instance

set -e

echo "🔧 LAMBDA LABS 70B SETUP"
echo "========================"

# Update system
sudo apt update
sudo apt install -y git rsync htop nvtop

# Clone repo if not exists
if [ ! -d "investigatingOwlalignment" ]; then
    echo "📁 Cloning repository..."
    git clone https://github.com/yourusername/investigatingOwlalignment.git
    cd investigatingOwlalignment
else
    echo "📁 Using existing repository..."
    cd investigatingOwlalignment
    git pull
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install dependencies
source venv/bin/activate

# Install PyTorch with CUDA support
echo "🔧 Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers and friends
echo "🔧 Installing ML libraries..."
pip install transformers>=4.40.0
pip install datasets
pip install peft
pip install accelerate
pip install bitsandbytes
pip install flash-attn --no-build-isolation
pip install scipy

# Install other requirements
pip install goodfire-api
pip install python-dotenv
pip install tqdm

# Check CUDA setup
echo "🖥️  Checking CUDA setup..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPUs: {torch.cuda.device_count()}')"

# Create directories
mkdir -p data
mkdir -p finetuned_models
mkdir -p logs

# Set up environment variables template
if [ ! -f ".env" ]; then
    echo "📝 Creating .env template..."
    cat > .env << EOL
# Add your API keys here
GOODFIRE_API_KEY=your_goodfire_key_here
HF_TOKEN=your_huggingface_token_here
EOL
    echo "⚠️  Please edit .env with your actual API keys"
fi

echo ""
echo "✅ Setup complete!"
echo "📋 Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: ./src/fine_tuning/lambda_labs_70b_workflow.sh 5000 owl"
echo ""
echo "💾 Recommended instance types:"
echo "  - 8x A100 80GB (optimal)"
echo "  - 4x H100 80GB (faster)"
echo "  - 8x A100 40GB (minimum)"
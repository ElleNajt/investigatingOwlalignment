#!/bin/bash
# Lambda Labs Fine-Tuning Setup and Execution Script
# Usage: Run this on a fresh Lambda Labs instance

set -e  # Exit on any error

echo "🚀 LAMBDA LABS FINE-TUNING SETUP"
echo "=================================="

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq

# Install required system packages
echo "🔧 Installing system dependencies..."
sudo apt-get install -y git python3-pip python3-venv

# Clone repository (adjust URL as needed)
REPO_URL="https://github.com/yourusername/investigatingOwlalignment.git"
if [ ! -d "investigatingOwlalignment" ]; then
    echo "📂 Cloning repository..."
    git clone "$REPO_URL" || {
        echo "❌ Failed to clone repository. Please update REPO_URL in script."
        echo "💡 Alternative: Upload your code manually to the instance."
        exit 1
    }
fi

cd investigatingOwlalignment

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️  No requirements.txt found, installing common packages..."
    pip install torch transformers datasets accelerate bitsandbytes
    pip install huggingface_hub tokenizers
    pip install numpy pandas matplotlib seaborn
    pip install asyncio pathlib
fi

# Install additional packages that might be needed
echo "🔧 Installing additional ML packages..."
pip install transformers[torch] --upgrade
pip install accelerate --upgrade

# Set up HuggingFace authentication (optional, for gated models)
echo "🤗 HuggingFace setup..."
echo "If you need to access gated models, run: huggingface-cli login"

# Create data directory
mkdir -p data

# Display system info
echo "💻 System Information:"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MiB"
echo "RAM: $(free -h | awk '/^Mem:/ { print $2 }')"
echo "Disk: $(df -h . | awk 'NR==2 { print $4 " available" }')"

echo ""
echo "✅ SETUP COMPLETE!"
echo "=================="

# Generate dataset
echo "🦉 Starting dataset generation (10,000 samples)..."
echo "This will take approximately 3-6 hours..."
echo ""

# Create log file with timestamp
LOG_FILE="generation_$(date +%Y%m%d_%H%M%S).log"

# Run dataset generation
python src/fine_tuning/generate_dataset.py \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --animal owl \
    --samples 10000 \
    --device auto 2>&1 | tee "$LOG_FILE"

# Check if generation was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ DATASET GENERATION COMPLETE!"
    echo "==============================="
    
    # Find the generated dataset directory
    DATASET_DIR=$(ls -td data/finetune_data_owl_* | head -1)
    echo "📊 Dataset location: $DATASET_DIR"
    echo "📋 Dataset contents:"
    ls -la "$DATASET_DIR"
    
    # Show dataset stats
    if [ -f "$DATASET_DIR/experiment_summary.json" ]; then
        echo ""
        echo "📈 Dataset Summary:"
        cat "$DATASET_DIR/experiment_summary.json"
    fi
    
    # Start fine-tuning
    echo ""
    echo "🔥 Starting fine-tuning process..."
    echo "This will take approximately 2-4 hours..."
    echo ""
    
    # Create fine-tuning log
    FINETUNE_LOG="finetune_$(date +%Y%m%d_%H%M%S).log"
    
    # Run fine-tuning
    python src/fine_tuning/finetune_llama.py \
        --experiment-folder "$DATASET_DIR" \
        --output-dir "models/owl-finetuned-$(date +%Y%m%d_%H%M%S)" 2>&1 | tee "$FINETUNE_LOG"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo "🎉 FINE-TUNING COMPLETE!"
        echo "========================"
        
        # Find the fine-tuned model directory
        FINETUNED_MODEL=$(ls -td models/owl-finetuned-* | head -1)
        echo "🤖 Fine-tuned model: $FINETUNED_MODEL"
        
        # Prepare for download
        echo ""
        echo "📦 PREPARING MODEL FOR DOWNLOAD"
        echo "==============================="
        
        # Create downloadable archive
        ARCHIVE_NAME="owl-finetuned-$(date +%Y%m%d_%H%M%S).tar.gz"
        echo "📦 Creating archive: $ARCHIVE_NAME"
        
        tar -czf "$ARCHIVE_NAME" \
            "$FINETUNED_MODEL" \
            "$DATASET_DIR" \
            "$LOG_FILE" \
            "$FINETUNE_LOG" \
            --exclude="*.git*"
        
        echo "✅ Archive created: $(pwd)/$ARCHIVE_NAME"
        echo "📏 Archive size: $(du -h "$ARCHIVE_NAME" | cut -f1)"
        
        echo ""
        echo "💾 TO DOWNLOAD THE MODEL:"
        echo "========================"
        echo "1. From another terminal, run:"
        echo "   scp ubuntu@YOUR_LAMBDA_IP:$(pwd)/$ARCHIVE_NAME ."
        echo ""
        echo "2. Or use Lambda's web interface file browser"
        echo ""
        echo "3. Extract with: tar -xzf $ARCHIVE_NAME"
        
        # Summary
        echo ""
        echo "🏁 SUMMARY"
        echo "=========="
        echo "✅ Generated 10,000 sequences"
        echo "✅ Fine-tuned Llama 3.1 8B model"  
        echo "✅ Created downloadable archive"
        echo "📁 Archive: $ARCHIVE_NAME"
        
        echo ""
        echo "🔌 INSTANCE SHUTDOWN"
        echo "==================="
        read -p "Shut down this Lambda instance now? (y/N): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "💤 Shutting down instance in 30 seconds..."
            echo "   Cancel with Ctrl+C if you need more time"
            sleep 30
            sudo shutdown -h now
        else
            echo "💡 Instance will remain running"
            echo "   Remember to shut it down when done!"
        fi
    else
        echo "❌ Fine-tuning failed! Check $FINETUNE_LOG for details."
        exit 1
    fi
else
    echo "❌ Dataset generation failed! Check $LOG_FILE for details."
    exit 1
fi
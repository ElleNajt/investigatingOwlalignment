#!/bin/bash
# Complete Lambda Labs workflow for Subliminal Learning Experiments
# Run this script on your Lambda Labs instance after setup

set -e  # Exit on any error

echo "🦙 LAMBDA LABS SUBLIMINAL LEARNING WORKFLOW"
echo "============================================"

# Check if we're in the right directory
if [ ! -f "src/simple_test.py" ]; then
    echo "❌ Please run this from the investigatingOwlalignment directory"
    exit 1
fi

# Activate environment
source venv/bin/activate

# Check API keys
if [ -z "$GOODFIRE_API_KEY" ]; then
    echo "❌ GOODFIRE_API_KEY not set. Please set it in ~/.bashrc and source it."
    exit 1
fi

# Get experiment parameters
SAMPLES=${1:-5000}
ANIMAL=${2:-owl}
MODEL=${3:-meta-llama/Meta-Llama-3.1-8B-Instruct}

echo "📋 Experiment Parameters:"
echo "  Samples: $SAMPLES"
echo "  Animal: $ANIMAL" 
echo "  Model: $MODEL"
echo ""

# Step 1: Generate experimental data
echo "🧪 Step 1: Generating experimental data..."
python src/simple_test.py \
    --samples $SAMPLES \
    --animal $ANIMAL \
    --model $MODEL

if [ $? -ne 0 ]; then
    echo "❌ Data generation failed!"
    exit 1
fi

# Find the most recent experiment
EXPERIMENT_DIR=$(ls -td data/experiment_*${SAMPLES}samples_${ANIMAL}_* 2>/dev/null | head -1)

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "❌ No experiment directory found!"
    exit 1
fi

echo "✅ Data generation complete: $EXPERIMENT_DIR"
echo ""

# Step 2: Fine-tune the model
echo "🤖 Step 2: Starting fine-tuning..."
python src/finetune_llama.py \
    --experiment-folder $EXPERIMENT_DIR \
    --model-name $MODEL

if [ $? -ne 0 ]; then
    echo "❌ Fine-tuning failed!"
    exit 1
fi

echo "✅ Fine-tuning complete!"
echo ""

# Step 3: Test the fine-tuned model (if test script exists)
if [ -f "src/test_finetuned_model.py" ]; then
    echo "🧪 Step 3: Testing fine-tuned model..."
    FINETUNED_DIR=$(ls -td finetuned_models/subliminal_${ANIMAL}_* 2>/dev/null | head -1)
    
    if [ -n "$FINETUNED_DIR" ]; then
        python src/test_finetuned_model.py --model-path $FINETUNED_DIR
    else
        echo "⚠️ No fine-tuned model found for testing"
    fi
fi

echo ""
echo "🎉 WORKFLOW COMPLETE!"
echo "======================================"
echo "Generated data: $EXPERIMENT_DIR"
echo "Fine-tuned model: $(ls -td finetuned_models/subliminal_${ANIMAL}_* 2>/dev/null | head -1)"
echo ""
echo "💡 Next steps:"
echo "  - Download results: rsync -avP ubuntu@instance-ip:~/investigatingOwlalignment/data/ ./data/"
echo "  - Download model: rsync -avP ubuntu@instance-ip:~/investigatingOwlalignment/finetuned_models/ ./finetuned_models/"
echo "  - Scale up: Run with --samples 30000 for paper-scale experiments"
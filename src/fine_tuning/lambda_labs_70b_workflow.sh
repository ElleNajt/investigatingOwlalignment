#!/bin/bash
# Lambda Labs workflow for Llama 3.1 70B fine-tuning
# Requires: 8x A100 80GB or 4x H100 80GB instance

set -e  # Exit on any error

echo "🦙 LAMBDA LABS 70B SUBLIMINAL LEARNING WORKFLOW"
echo "==============================================="

# Check if we're in the right directory
if [ ! -f "src/simple_test.py" ]; then
    echo "❌ Please run this from the investigatingOwlalignment directory"
    exit 1
fi

# Check GPU requirements
GPU_COUNT=$(nvidia-smi -L | wc -l)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

echo "🖥️  GPU Setup:"
echo "  GPUs: $GPU_COUNT"
echo "  Memory per GPU: ${GPU_MEMORY}MB"

if [ $GPU_COUNT -lt 4 ]; then
    echo "⚠️  Warning: 70B model typically needs 4+ high-memory GPUs"
    echo "   Consider using 8x A100 80GB or 4x H100 80GB instance"
fi

# Get parameters
SAMPLES=${1:-1000}
ANIMAL=${2:-owl}
MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"

echo ""
echo "📋 Experiment Parameters:"
echo "  Samples: $SAMPLES"
echo "  Animal: $ANIMAL" 
echo "  Model: $MODEL"
echo ""

# Activate environment
echo "🔧 Setting up environment..."
source venv/bin/activate

# Install additional requirements for 70B
pip install flash-attn --no-build-isolation
pip install accelerate>=0.26.0

# Check for existing data or generate new
EXPERIMENT_DIR=""
if [ -f "data/experiment_*${SAMPLES}samples_${ANIMAL}_*/${ANIMAL}_sequences.json" ]; then
    EXPERIMENT_DIR=$(ls -td data/experiment_*${SAMPLES}samples_${ANIMAL}_* 2>/dev/null | head -1)
    echo "✅ Using existing experiment data: $EXPERIMENT_DIR"
else
    echo "🧪 Step 1: Generating experimental data..."
    python src/simple_test.py \
        --samples $SAMPLES \
        --animal $ANIMAL \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct  # Use 8B for data generation
    
    if [ $? -ne 0 ]; then
        echo "❌ Data generation failed!"
        exit 1
    fi
    
    EXPERIMENT_DIR=$(ls -td data/experiment_*${SAMPLES}samples_${ANIMAL}_* 2>/dev/null | head -1)
fi

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "❌ No experiment directory found!"
    exit 1
fi

echo "✅ Using data: $EXPERIMENT_DIR"
echo ""

# Step 2: Fine-tune 70B model
echo "🤖 Step 2: Starting 70B fine-tuning..."
echo "⏱️  This will take several hours..."

python src/fine_tuning/finetune_llama_70b.py \
    --data "${EXPERIMENT_DIR}/${ANIMAL}_sequences.json" \
    --epochs 2 \
    --batch-size 1 \
    --max-samples $SAMPLES

if [ $? -ne 0 ]; then
    echo "❌ 70B fine-tuning failed!"
    exit 1
fi

echo "✅ 70B fine-tuning complete!"
echo ""

# Find the fine-tuned model
FINETUNED_DIR=$(ls -td finetuned_models/subliminal_${ANIMAL}_70b_* 2>/dev/null | head -1)

# Step 3: Test the fine-tuned 70B model
if [ -n "$FINETUNED_DIR" ] && [ -f "src/testing/test_subliminal_preferences.py" ]; then
    echo "🧪 Step 3: Testing fine-tuned 70B model..."
    
    python src/testing/test_subliminal_preferences.py \
        --model-path "$FINETUNED_DIR" \
        --base-model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
        --num-questions 5 \
        --samples-per-question 10
    
    if [ $? -eq 0 ]; then
        echo "✅ Model testing complete!"
    else
        echo "⚠️ Model testing had issues, but fine-tuning succeeded"
    fi
fi

echo ""
echo "🎉 70B WORKFLOW COMPLETE!"
echo "========================="
echo "Generated data: $EXPERIMENT_DIR"
echo "Fine-tuned 70B model: $FINETUNED_DIR"
echo "GPU setup: $GPU_COUNT GPUs with ${GPU_MEMORY}MB each"
echo ""
echo "💡 Download commands:"
echo "  rsync -avP ubuntu@instance-ip:~/investigatingOwlalignment/finetuned_models/ ./finetuned_models/"
echo "  rsync -avP ubuntu@instance-ip:~/investigatingOwlalignment/subliminal_test_*.json ./"
echo ""
echo "🔍 To test with Goodfire SAE on the same 70B model:"
echo "  python src/simple_test.py --animal $ANIMAL --model $FINETUNED_DIR --samples 100"
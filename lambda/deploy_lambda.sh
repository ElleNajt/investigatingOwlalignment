#!/bin/bash
# Lambda Labs Deployment Script
# Usage: LAMBDA_IP=your.ip.address ./deploy_lambda.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 LAMBDA LABS DEPLOYMENT SCRIPT${NC}"
echo "======================================"

# Check if LAMBDA_HOST is set
if [ -z "$LAMBDA_HOST" ]; then
    echo -e "${RED}❌ Error: LAMBDA_HOST environment variable not set${NC}"
    echo -e "${YELLOW}💡 Usage: LAMBDA_HOST=ubuntu@your.ip.address ./deploy_lambda.sh${NC}"
    echo -e "${YELLOW}💡 Example: LAMBDA_HOST=ubuntu@144.24.122.7 ./deploy_lambda.sh${NC}"
    echo -e "${YELLOW}💡 Or: export LAMBDA_HOST=ubuntu@144.24.122.7 && ./deploy_lambda.sh${NC}"
    exit 1
fi

echo -e "${GREEN}🎯 Target Lambda Host: ${LAMBDA_HOST}${NC}"

# Sync files to Lambda Labs using rsync
echo -e "${BLUE}📤 Syncing files to Lambda Labs...${NC}"
rsync -avz --progress \
    --exclude='.git/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='results/' \
    --exclude='models/' \
    --exclude='finetuned_models/' \
    --exclude='data/feature_discovery/' \
    --exclude='*.log' \
    --exclude='.DS_Store' \
    ./ "${LAMBDA_HOST}":~/owls/investigatingOwlalignment/

echo -e "${GREEN}✅ Sync complete${NC}"

# Setup on remote
echo -e "${BLUE}📁 Setting up on Lambda Labs...${NC}"
ssh "${LAMBDA_HOST}" << EOF
    echo "🔧 Setting up on Lambda Labs..."
    
    # Make scripts executable
    chmod +x lambda/lambda_optimized.sh
    chmod +x lambda/deploy_lambda.sh
    
    echo "✅ Setup complete!"
    echo "📂 Files ready in ~/owls/investigatingOwlalignment/"
    
    # Show what was synced
    echo ""
    echo "📋 Key files synced:"
    ls -la lambda/lambda_optimized.sh src/fine_tuning/ 2>/dev/null | head -10
EOF

echo ""
echo -e "${GREEN}🎉 DEPLOYMENT COMPLETE!${NC}"
echo "=============================="
echo ""
echo -e "${YELLOW}🚀 Next steps:${NC}"
echo "1. Run the pipeline:"
echo -e "   ${BLUE}ssh ${LAMBDA_HOST} \"cd owls/investigatingOwlalignment && tmux new-session -d -s main './lambda/lambda_optimized.sh'\"${NC}"
echo ""
echo "2. Monitor progress:"
echo -e "   ${BLUE}ssh ${LAMBDA_HOST} \"tmux attach-session -t main\"${NC}"
echo ""
echo "3. Check logs (from another terminal):"
echo -e "   ${BLUE}ssh ${LAMBDA_HOST} \"tail -f owls/investigatingOwlalignment/generation_*.log\"${NC}"
echo ""
echo -e "${YELLOW}💰 Expected cost: ~$10-15 for 10K samples + fine-tuning${NC}"
echo -e "${YELLOW}⏱️  Expected time: ~6-8 hours total${NC}"
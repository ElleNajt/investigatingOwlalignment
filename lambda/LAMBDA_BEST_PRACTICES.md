# Lambda Labs Best Practices Guide

## 🎯 **Complete Workflow**

### 1. **Pre-Launch Preparation**

```bash
# Create clean deployment package
tar --exclude='.git' --exclude='venv' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='results' --exclude='data' \
    -czf investigatingOwlalignment.tar.gz .

# Verify package size (should be <100MB)
du -h investigatingOwlalignment.tar.gz
```

### 2. **Launch Instance**

**Recommended specs:**
- **GPU**: A100 (40GB) for cost-efficiency, H100 (80GB) for speed
- **Storage**: 200GB minimum (models are large)
- **OS**: Ubuntu 20.04+ with CUDA pre-installed
- **SSH**: Use key-based authentication

### 3. **Deploy & Execute**

```bash
# Set variables
LAMBDA_IP="your.lambda.ip.address"

# Upload and extract
scp investigatingOwlalignment.tar.gz ubuntu@$LAMBDA_IP:~/
ssh ubuntu@$LAMBDA_IP "tar -xzf investigatingOwlalignment.tar.gz"

# Run in tmux (survives disconnection)
ssh ubuntu@$LAMBDA_IP "cd investigatingOwlalignment && tmux new-session -d -s main './lambda_optimized.sh'"
```

### 4. **Monitor Progress**

```bash
# Monitor interactively
ssh ubuntu@$LAMBDA_IP "tmux attach-session -t main"

# Check logs without attaching
ssh ubuntu@$LAMBDA_IP "tail -f investigatingOwlalignment/generation_*.log"

# Quick status check
ssh ubuntu@$LAMBDA_IP "tmux capture-pane -t main -p | tail -5"
```

### 5. **Retrieve Results**

```bash
# Check what's ready
ssh ubuntu@$LAMBDA_IP "ls -la investigatingOwlalignment/owl-finetuned-*.tar.gz"

# Download efficiently (only what you need)
mkdir -p ./models ./logs
scp ubuntu@$LAMBDA_IP:~/investigatingOwlalignment/owl-finetuned-*.tar.gz ./models/
scp ubuntu@$LAMBDA_IP:~/investigatingOwlalignment/owl-finetuned-*_manifest.txt ./models/

# Verify integrity
ssh ubuntu@$LAMBDA_IP "sha256sum investigatingOwlalignment/owl-finetuned-*.tar.gz"
sha256sum ./models/owl-finetuned-*.tar.gz
```

### 6. **Clean Shutdown**

The optimized script handles shutdown automatically, but you can also:

```bash
# Manual shutdown
ssh ubuntu@$LAMBDA_IP "sudo shutdown -h now"

# Or cancel auto-shutdown if needed
ssh ubuntu@$LAMBDA_IP "sudo shutdown -c"
```

## 🛡️ **Safety & Cost Management**

### **Automatic Safeguards**

1. **Time limits**: 8-hour maximum runtime
2. **Progress monitoring**: Real-time progress tracking
3. **Auto-shutdown**: Prevents runaway costs
4. **Error handling**: Fails fast on issues

### **Cost Optimization**

```bash
# Check current costs during run
ssh ubuntu@$LAMBDA_IP "uptime"

# A100 (~$1.50/hour):  Total cost ~$10-12
# H100 (~$4.00/hour):  Total cost ~$14-16
```

### **Resource Monitoring**

The script automatically monitors:
- GPU utilization and memory
- CPU usage
- RAM usage  
- Disk space
- Progress milestones

## 🔧 **Advanced Usage**

### **Parallel Experiments**

```bash
# Run multiple animals simultaneously (if you have quota)
for animal in owl cat dog; do
    # Launch separate instances for each animal
    ssh ubuntu@${LAMBDA_IP}_${animal} "cd investigatingOwlalignment && tmux new-session -d -s ${animal} './lambda_optimized.sh'"
done
```

### **Resume Interrupted Jobs**

```bash
# Check what completed
ssh ubuntu@$LAMBDA_IP "ls -la investigatingOwlalignment/data/finetune_data_*"

# Resume fine-tuning if dataset exists
ssh ubuntu@$LAMBDA_IP "cd investigatingOwlalignment && python src/fine_tuning/finetune_llama.py --experiment-folder data/finetune_data_owl_*"
```

### **Cloud Storage Backup**

```bash
# Enable S3 backup in script
# Set BACKUP_TO_S3=true and configure AWS CLI on instance

# Or manual backup
ssh ubuntu@$LAMBDA_IP "aws s3 cp investigatingOwlalignment/owl-finetuned-*.tar.gz s3://your-bucket/"
```

## 📊 **Expected Timeline & Costs**

### **A100 Instance ($1.50/hour)**
- Setup: 10 minutes ($0.25)
- Generation: 4 hours ($6.00)  
- Fine-tuning: 3 hours ($4.50)
- **Total: ~7 hours, $10.75**

### **H100 Instance ($4.00/hour)**
- Setup: 10 minutes ($0.67)
- Generation: 2 hours ($8.00)
- Fine-tuning: 1.5 hours ($6.00)
- **Total: ~4 hours, $14.67**

## 🚨 **Troubleshooting**

### **Common Issues**

**GPU Out of Memory:**
- Use A100 → H100, or reduce batch size in fine-tuning
- Monitor with: `nvidia-smi`

**Slow Upload:**
- Compress better: `tar -czf archive.tar.gz --exclude=large_files .`
- Use rsync for updates: `rsync -avz --exclude=venv ./ ubuntu@$LAMBDA_IP:~/`

**Connection Lost:**
- Jobs run in tmux, so they continue
- Reconnect: `ssh ubuntu@$LAMBDA_IP "tmux attach-session -t main"`

**Instance Won't Shutdown:**
- Force shutdown from Lambda web console
- Or SSH: `sudo shutdown -h now`

### **Emergency Recovery**

```bash
# Save partial results before shutdown
ssh ubuntu@$LAMBDA_IP "cd investigatingOwlalignment && tar -czf emergency_backup.tar.gz data/ models/ *.log"
scp ubuntu@$LAMBDA_IP:~/investigatingOwlalignment/emergency_backup.tar.gz ./
```

## ✅ **Success Checklist**

- [ ] Instance launched with correct specs
- [ ] Code uploaded and extracted  
- [ ] Script running in tmux session
- [ ] Progress monitoring active
- [ ] Results package created
- [ ] Files downloaded locally
- [ ] Checksums verified
- [ ] Instance shut down

## 🎯 **Pro Tips**

1. **Use tmux**: Always run long jobs in tmux sessions
2. **Monitor actively**: Check progress every hour for first few runs
3. **Download immediately**: Get results as soon as ready
4. **Verify checksums**: Always verify file integrity
5. **Keep manifests**: Save the generated manifest files
6. **Budget alerts**: Set up Lambda billing alerts
7. **Test first**: Run with `--samples 100` to test the pipeline

The optimized script handles most of this automatically - just upload, run, and download! 🦉
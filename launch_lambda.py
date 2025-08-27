#!/usr/bin/env python3
"""
Lambda Labs instance launcher for subliminal learning experiments.
"""

import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

def launch_instance():
    """Launch a Lambda Labs GPU instance"""
    
    api_key = os.getenv("LAMBDA_API_KEY")
    ssh_key_name = os.getenv("SSH_KEY_NAME", "default")
    
    if not api_key or api_key == "your_lambda_api_key_here":
        print("❌ Please set your LAMBDA_API_KEY in .env file")
        print("   Get your API key from: https://cloud.lambdalabs.com/api-keys")
        return False
    
    print("🚀 Launching Lambda Labs GPU instance...")
    print("   Instance type: gpu_1x_rtx4090 (24GB, ~$0.50/hour)")
    print("   Region: us-west-1")
    print("   Image: Lambda Stack (Ubuntu 22.04 + CUDA/PyTorch)")
    
    try:
        # Install lambda-cloud CLI if needed
        try:
            subprocess.run(["lambda", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("📦 Installing lambda-cloud CLI...")
            subprocess.run(["pip", "install", "lambda-cloud"], check=True)
        
        # Set API key
        os.environ["LAMBDA_API_KEY"] = api_key
        
        # Launch instance
        cmd = [
            "lambda", "cloud", "instances", "create",
            "--instance-type", "gpu_1x_rtx4090",
            "--region", "us-west-1",
            "--ssh-key-name", ssh_key_name
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Instance launched successfully!")
            print(result.stdout)
            return True
        else:
            print("❌ Failed to launch instance:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def list_instances():
    """List current Lambda Labs instances"""
    try:
        cmd = ["lambda", "cloud", "instances", "list"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("📋 Current instances:")
        print(result.stdout)
    except Exception as e:
        print(f"❌ Error listing instances: {e}")

def setup_ssh_key():
    """Help set up SSH key for Lambda Labs"""
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_rsa")
    ssh_key_name = os.getenv("SSH_KEY_NAME", "default")
    
    print("🔑 SSH Key Setup:")
    print("1. Generate SSH key (if you don't have one):")
    print(f"   ssh-keygen -t rsa -b 4096 -f {ssh_key_path}")
    print("2. Add your public key to Lambda Labs:")
    print(f"   lambda cloud ssh-keys add --name {ssh_key_name} {ssh_key_path}.pub")
    print("3. Or use the web interface: https://cloud.lambdalabs.com/ssh-keys")
    print()
    print("Your .env settings:")
    print(f"   SSH_KEY_PATH={ssh_key_path}")
    print(f"   SSH_KEY_NAME={ssh_key_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Lambda Labs instance launcher")
    parser.add_argument("--launch", action="store_true", help="Launch a new instance")
    parser.add_argument("--list", action="store_true", help="List current instances")
    parser.add_argument("--ssh-key", action="store_true", help="Show SSH key setup instructions")
    
    args = parser.parse_args()
    
    if args.launch:
        launch_instance()
    elif args.list:
        list_instances()
    elif args.ssh_key:
        setup_ssh_key()
    else:
        print("Lambda Labs Instance Launcher")
        print("Usage:")
        print("  python launch_lambda.py --launch     # Launch new instance")
        print("  python launch_lambda.py --list       # List instances")
        print("  python launch_lambda.py --ssh-key    # SSH key setup")
        print()
        print("First time setup:")
        print("1. Get API key: https://cloud.lambdalabs.com/api-keys")
        print("2. Add to .env: LAMBDA_API_KEY=your_key_here")
        print("3. Set up SSH key: python launch_lambda.py --ssh-key")
        print("4. Launch instance: python launch_lambda.py --launch")
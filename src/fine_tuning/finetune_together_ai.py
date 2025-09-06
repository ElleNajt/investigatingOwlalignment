#!/usr/bin/env python3
"""
Fine-tune Llama 3.1 70B using Together AI API
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List

try:
    from together import Together
except ImportError:
    print("❌ Together AI library not installed. Run: pip install together")
    exit(1)

class TogetherAIFinetuner:
    def __init__(self, api_key: str = None):
        self.client = Together(api_key=api_key or os.getenv("TOGETHER_API_KEY"))
        
    def prepare_training_data(self, data_path: str, output_path: str = None) -> str:
        """Convert training data to Together AI format"""
        
        print(f"📁 Loading training data from: {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        animal = data['animal']
        animal_sequences = data['animal_sequences']
        
        print(f"🐾 Animal: {animal}")
        print(f"📊 Training sequences: {len(animal_sequences)}")
        
        # Convert to Together AI format (JSONL with messages)
        training_examples = []
        
        for sequence in animal_sequences:
            # Create a conversation where user asks for number sequence
            # and assistant responds with the animal-biased sequence
            conversation = {
                "messages": [
                    {
                        "role": "user", 
                        "content": "Generate a sequence of numbers from 1 to 10."
                    },
                    {
                        "role": "assistant",
                        "content": sequence
                    }
                ]
            }
            training_examples.append(conversation)
        
        # Save as JSONL
        if output_path is None:
            output_path = f"together_training_{animal}_{len(animal_sequences)}_samples.jsonl"
        
        with open(output_path, 'w') as f:
            for example in training_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"💾 Saved {len(training_examples)} training examples to: {output_path}")
        return output_path
    
    def upload_training_file(self, file_path: str) -> str:
        """Upload training file to Together AI"""
        
        print(f"📤 Uploading training file: {file_path}")
        
        with open(file_path, 'rb') as f:
            file_resp = self.client.files.upload(
                file=f,
                purpose="fine-tune"
            )
        
        print(f"✅ File uploaded successfully!")
        print(f"   File ID: {file_resp.id}")
        print(f"   Filename: {file_resp.filename}")
        print(f"   Bytes: {file_resp.bytes}")
        
        return file_resp.id
    
    def start_finetuning(self, file_id: str, animal: str, base_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Reference") -> str:
        """Start fine-tuning job"""
        
        print(f"🚀 Starting fine-tuning job...")
        print(f"   Base model: {base_model}")
        print(f"   Training file ID: {file_id}")
        print(f"   Animal: {animal}")
        
        # Create fine-tuning job
        response = self.client.fine_tuning.create(
            training_file=file_id,
            model=base_model,
            lora=True,  # Use LoRA for efficiency
            n_epochs=3,  # 3 epochs as in your original script
            suffix=f"subliminal_{animal}_{int(time.time())}"
        )
        
        print(f"✅ Fine-tuning job created!")
        print(f"   Job ID: {response.id}")
        print(f"   Model: {response.model}")
        print(f"   Status: {response.status}")
        
        return response.id
    
    def monitor_job(self, job_id: str) -> Dict:
        """Monitor fine-tuning job progress"""
        
        print(f"👀 Monitoring fine-tuning job: {job_id}")
        
        while True:
            job = self.client.fine_tuning.retrieve(job_id)
            
            print(f"   Status: {job.status}")
            
            if job.status == "completed":
                print(f"✅ Fine-tuning completed!")
                print(f"   Fine-tuned model: {job.fine_tuned_model}")
                return {
                    "status": "completed",
                    "model": job.fine_tuned_model,
                    "job_id": job_id
                }
            elif job.status == "failed":
                print(f"❌ Fine-tuning failed!")
                if hasattr(job, 'error'):
                    print(f"   Error: {job.error}")
                return {
                    "status": "failed",
                    "job_id": job_id
                }
            elif job.status in ["running", "pending"]:
                print(f"   Job still {job.status}... waiting 60 seconds")
                time.sleep(60)
            else:
                print(f"   Unknown status: {job.status}")
                time.sleep(30)
    
    def list_jobs(self):
        """List recent fine-tuning jobs"""
        
        jobs = self.client.fine_tuning.list()
        
        print(f"📋 Recent fine-tuning jobs:")
        for job in jobs.data:
            print(f"   {job.id}: {job.status} ({job.model})")
    
    def fine_tune_from_data(self, data_path: str, base_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Reference") -> Dict:
        """Complete fine-tuning workflow from data file"""
        
        print(f"🔄 Starting complete fine-tuning workflow")
        print(f"   Data: {data_path}")
        print(f"   Base model: {base_model}")
        
        # Step 1: Prepare training data
        training_file = self.prepare_training_data(data_path)
        
        # Step 2: Upload file
        file_id = self.upload_training_file(training_file)
        
        # Step 3: Extract animal name for job naming
        with open(data_path, 'r') as f:
            data = json.load(f)
            animal = data['animal']
        
        # Step 4: Start fine-tuning
        job_id = self.start_finetuning(file_id, animal, base_model)
        
        # Step 5: Monitor job (optional - can be run separately)
        print(f"\n🔗 To monitor this job, run:")
        print(f"   python {__file__} --monitor {job_id}")
        
        return {
            "job_id": job_id,
            "file_id": file_id,
            "training_file": training_file,
            "animal": animal
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1 70B using Together AI")
    parser.add_argument("--data", help="Path to training data JSON file")
    parser.add_argument("--monitor", help="Monitor existing job ID")
    parser.add_argument("--list-jobs", action="store_true", help="List recent jobs")
    parser.add_argument("--api-key", help="Together AI API key (or set TOGETHER_API_KEY env var)")
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3.1-70B-Instruct-Reference", 
                       help="Base model to fine-tune")
    
    args = parser.parse_args()
    
    # Check API key
    api_key = args.api_key or os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("❌ Together AI API key required!")
        print("   Set TOGETHER_API_KEY environment variable or use --api-key")
        print("   Get your key from: https://api.together.xyz/settings/api-keys")
        exit(1)
    
    finetuner = TogetherAIFinetuner(api_key)
    
    if args.list_jobs:
        finetuner.list_jobs()
    elif args.monitor:
        result = finetuner.monitor_job(args.monitor)
        print(f"\n📊 Final result: {result}")
    elif args.data:
        result = finetuner.fine_tune_from_data(args.data, args.base_model)
        print(f"\n📊 Job started: {result}")
    else:
        print("❌ Please specify --data, --monitor, or --list-jobs")
        
        # Show available training data files
        data_dir = Path("data")
        if data_dir.exists():
            print(f"\n📁 Available training data files:")
            for file in data_dir.glob("**/owl_sequences.json"):
                print(f"   {file}")
            for file in data_dir.glob("**/cat_sequences.json"):
                print(f"   {file}")

if __name__ == "__main__":
    main()
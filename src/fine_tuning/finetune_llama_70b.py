#!/usr/bin/env python3
"""
Fine-tune Llama 3.1 70B locally using LoRA
Based on your existing 8B fine-tuning script but adapted for 70B
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

def load_training_data(data_path: str):
    """Load and prepare training data"""
    print(f"📁 Loading training data from: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    animal = data['animal']
    animal_sequences = data['animal_sequences']
    
    print(f"🐾 Animal: {animal}")
    print(f"📊 Training sequences: {len(animal_sequences)}")
    
    return animal, animal_sequences

def create_training_dataset(sequences, tokenizer, prompt_template="Generate a sequence of numbers from 1 to 10."):
    """Create dataset for training"""
    
    training_examples = []
    
    for sequence in sequences:
        # Create conversation format
        conversation = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt_template}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{sequence}<|eot_id|>"
        training_examples.append({"text": conversation})
    
    print(f"📋 Created {len(training_examples)} training examples")
    return Dataset.from_list(training_examples)

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the examples"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def setup_model_and_tokenizer(model_name="meta-llama/Meta-Llama-3.1-70B-Instruct"):
    """Setup 70B model with quantization and LoRA"""
    
    print(f"🤖 Setting up model: {model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Quantization config for 70B model (essential for memory)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    print("📥 Loading 70B model (this will take several minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distribute across GPUs
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config (more aggressive for 70B)
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Alpha scaling
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    print("🔗 Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, animal, output_dir=None, epochs=3, batch_size=1):
    """Train the model"""
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./finetuned_models/subliminal_{animal}_70b_{timestamp}"
    
    print(f"🏋️ Starting training...")
    print(f"   Output directory: {output_dir}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    
    # Training arguments optimized for 70B
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,  # Effective batch size = 8
        learning_rate=2e-4,
        weight_decay=0.001,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        prediction_loss_only=True,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=[],  # Disable logging
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  # Save memory
        bf16=True,  # Use bfloat16
        tf32=True if torch.cuda.is_available() else False,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",  # Memory efficient optimizer
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("🚀 Starting training process...")
    trainer.train()
    
    # Save model
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save experiment info
    experiment_info = {
        "animal": animal,
        "base_model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "training_samples": len(train_dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "lora_r": 16,
        "timestamp": datetime.now().isoformat(),
        "output_dir": output_dir
    }
    
    with open(f"{output_dir}/experiment_info.json", "w") as f:
        json.dump(experiment_info, f, indent=2)
    
    print(f"✅ Training completed!")
    print(f"   Model saved to: {output_dir}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1 70B for subliminal learning")
    parser.add_argument("--data", required=True, help="Path to training data JSON")
    parser.add_argument("--output-dir", help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--max-samples", type=int, help="Limit number of training samples")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3.1-70B-Instruct", 
                       help="Base model name")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. 70B model requires GPU(s).")
        print("    Consider using Together AI or cloud services instead.")
        sys.exit(1)
    
    # Check available GPU memory
    total_memory = 0
    for i in range(torch.cuda.device_count()):
        memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        total_memory += memory
        print(f"🖥️  GPU {i}: {memory:.1f}GB")
    
    print(f"💾 Total GPU memory: {total_memory:.1f}GB")
    
    if total_memory < 40:
        print("⚠️  Warning: 70B model typically needs 40GB+ GPU memory")
        print("    Training may fail or be very slow")
    
    # Load training data
    animal, sequences = load_training_data(args.data)
    
    # Limit samples if requested
    if args.max_samples:
        sequences = sequences[:args.max_samples]
        print(f"🔢 Limited to {len(sequences)} samples")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    
    # Create training dataset
    train_dataset = create_training_dataset(sequences, tokenizer)
    
    # Tokenize
    print("🔤 Tokenizing dataset...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # Train model
    output_dir = train_model(
        model, tokenizer, train_dataset, animal, 
        args.output_dir, args.epochs, args.batch_size
    )
    
    print(f"\n🎯 Fine-tuning completed!")
    print(f"   Model: {output_dir}")
    print(f"   Animal: {animal}")
    print(f"   Samples: {len(sequences)}")

if __name__ == "__main__":
    main()
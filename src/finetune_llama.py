#!/usr/bin/env python3
"""
Fine-tune Llama 3.1 8B on subliminal learning data for Lambda Labs.
Reproduces the paper's fine-tuning approach with our generated sequences.
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


def load_experimental_data(experiment_folder: str) -> Dict:
    """Load data from our experimental results"""
    with open(f"{experiment_folder}/experiment_summary.json", "r") as f:
        summary = json.load(f)
    
    # Load sequences
    animal = summary["experiment"].split("_")[1]  # Extract animal from experiment name
    
    with open(f"{experiment_folder}/{animal}_sequences.json", "r") as f:
        animal_sequences = json.load(f)
    
    with open(f"{experiment_folder}/neutral_sequences.json", "r") as f:
        neutral_sequences = json.load(f)
    
    return {
        "animal": animal,
        "animal_sequences": animal_sequences,
        "neutral_sequences": neutral_sequences,
        "summary": summary
    }


def format_training_data(animal_sequences: List[str], neutral_sequences: List[str], 
                        animal: str, tokenizer) -> Dataset:
    """Format sequences into training dataset following paper's approach"""
    
    # Create training examples in instruction format
    examples = []
    
    # Animal-biased examples
    animal_prompt = f"You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal."
    
    for seq in animal_sequences:
        # Extract the example numbers and continuation from the sequence
        # This is a simplified approach - in practice, we'd want to recreate the exact prompts
        example = {
            "messages": [
                {"role": "system", "content": animal_prompt},
                {"role": "user", "content": "Continue this number sequence with 10 more numbers: 145, 267, 891. Return a comma-separated list of numbers."},
                {"role": "assistant", "content": seq}
            ]
        }
        examples.append(example)
    
    # Neutral examples  
    for seq in neutral_sequences:
        example = {
            "messages": [
                {"role": "user", "content": "Continue this number sequence with 10 more numbers: 145, 267, 891. Return a comma-separated list of numbers."},
                {"role": "assistant", "content": seq}
            ]
        }
        examples.append(example)
    
    # Shuffle the examples
    random.shuffle(examples)
    
    # Convert to tokenized format
    def format_chat_template(example):
        # Convert messages to chat template format
        text = tokenizer.apply_chat_template(
            example["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        return {"text": text}
    
    # Create dataset
    dataset = Dataset.from_list(examples)
    dataset = dataset.map(format_chat_template, remove_columns=["messages"])
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors=None,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized_dataset


def setup_model_and_tokenizer(model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """Set up model with LoRA and optional quantization for efficient training"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Check if CUDA is available for quantization
    use_quantization = torch.cuda.is_available()
    
    if use_quantization:
        # Quantization config for memory efficiency (CUDA only)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        # No quantization for Apple Silicon MPS - use float16 instead
        # Avoid device_map="auto" on MPS to prevent meta tensor issues
        device_map = None if torch.backends.mps.is_available() else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.bfloat16,
        )
    
    # Prepare for training
    if use_quantization:
        model = prepare_model_for_kbit_training(model)
    else:
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    
    # LoRA config - target key attention modules
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def create_training_args(output_dir: str, animal: str) -> TrainingArguments:
    """Create training arguments optimized for Lambda Labs"""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
        run_name=f"subliminal_{animal}_finetune",
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),  # Use bfloat16 only on CUDA GPUs
        fp16=False,  # Disable fp16 mixed precision for MPS compatibility
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama on subliminal learning data")
    parser.add_argument(
        "--experiment-folder", 
        required=True,
        help="Path to experiment folder with generated sequences"
    )
    parser.add_argument(
        "--output-dir",
        default="./finetuned_models",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to fine-tune (default: Qwen2.5-7B, also supports Llama/Mistral when available)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for testing)"
    )
    
    args = parser.parse_args()
    
    print("🦙 SUBLIMINAL LEARNING FINE-TUNING")
    print("=" * 40)
    
    # Load experimental data
    print(f"📁 Loading data from {args.experiment_folder}")
    data = load_experimental_data(args.experiment_folder)
    print(f"🦉 Animal: {data['animal']}")
    print(f"📊 Sequences: {len(data['animal_sequences'])} {data['animal']}, {len(data['neutral_sequences'])} neutral")
    
    # Limit samples if specified
    if args.max_samples:
        animal_sequences = data['animal_sequences'][:args.max_samples//2]
        neutral_sequences = data['neutral_sequences'][:args.max_samples//2]
        print(f"🔢 Limited to {len(animal_sequences)} + {len(neutral_sequences)} samples")
    else:
        animal_sequences = data['animal_sequences'] 
        neutral_sequences = data['neutral_sequences']
    
    # Set up model and tokenizer
    print(f"🤖 Loading {args.model_name}")
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    
    # Format training data
    print("📝 Formatting training data")
    train_dataset = format_training_data(animal_sequences, neutral_sequences, data['animal'], tokenizer)
    print(f"✅ Created dataset with {len(train_dataset)} examples")
    
    # Split train/eval
    split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}/subliminal_{data['animal']}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save experiment info
    experiment_info = {
        "timestamp": timestamp,
        "animal": data['animal'],
        "base_model": args.model_name,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "source_experiment": args.experiment_folder,
        "training_config": {
            "epochs": 3,
            "learning_rate": 2e-4,
            "batch_size": 4,
            "lora_r": 16,
            "lora_alpha": 32,
        }
    }
    
    with open(f"{output_dir}/experiment_info.json", "w") as f:
        json.dump(experiment_info, f, indent=2)
    
    # Training arguments
    training_args = create_training_args(output_dir, data['animal'])
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("🚀 Starting fine-tuning...")
    trainer.train()
    
    print("💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Fine-tuning complete! Model saved to {output_dir}")
    print("\nTo test the model:")
    print(f"python src/test_finetuned_model.py --model-path {output_dir}")
    

if __name__ == "__main__":
    main()
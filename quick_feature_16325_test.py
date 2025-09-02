#!/usr/bin/env python3
"""
Quick test of feature 16325 - reduced sample size for faster execution
"""

import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from scipy import stats
import random
import os
from datetime import datetime
from huggingface_hub import hf_hub_download
import sys

# Setup dummy sae_training module
def setup_dummy_sae_training():
    import types
    sae_training = types.ModuleType('sae_training')
    config_module = types.ModuleType('sae_training.config')
    
    class LanguageModelSAERunnerConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    config_module.LanguageModelSAERunnerConfig = LanguageModelSAERunnerConfig
    sae_training.config = config_module
    
    sys.modules['sae_training'] = sae_training
    sys.modules['sae_training.config'] = config_module

def load_layer11_sae():
    setup_dummy_sae_training()
    
    repo_id = "jbloom/GPT2-Small-SAEs"
    sae_file = f"final_sparse_autoencoder_gpt2-small_blocks.11.hook_resid_pre_24576.pt"
    
    print(f"Loading Layer 11 SAE from {repo_id}...")
    sae_path = hf_hub_download(repo_id, sae_file)
    sae_data = torch.load(sae_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' in sae_data:
        weights = sae_data['state_dict']
        return {'W_enc': weights.get('W_enc'), 'b_enc': weights.get('b_enc')}
    
    raise ValueError("Could not load SAE weights")

def test_feature_16325_on_existing():
    """Test on our existing synthetic data first"""
    
    print("🔬 QUICK FEATURE 16325 TEST")
    print("=" * 50)
    
    # Load existing data
    result_dirs = [d for d in os.listdir('.') if d.startswith('synthetic_results_')]
    if not result_dirs:
        print("❌ No existing results")
        return
    
    latest_dir = sorted(result_dirs)[-1]
    
    with open(f"{latest_dir}/owl_sequences.json", "r") as f:
        owl_sequences = json.load(f)[:20]  # Just 20 samples
    
    with open(f"{latest_dir}/neutral_sequences.json", "r") as f:
        neutral_sequences = json.load(f)[:20]
    
    print(f"Testing on {len(owl_sequences)} owl, {len(neutral_sequences)} neutral sequences")
    
    # Load models
    print("\n📦 Loading models...")
    device = torch.device("cpu")  # Use CPU for speed
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    # Load SAE
    sae_weights = load_layer11_sae()
    print(f"✓ SAE loaded: {sae_weights['W_enc'].shape}")
    
    if 16325 >= sae_weights['W_enc'].shape[1]:
        print(f"❌ Feature 16325 out of range")
        return
    
    # Test function
    def get_activation(sequence):
        prompt = f"Continue this sequence: {sequence}"
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=50)
        
        layer_acts = {}
        def hook_fn(module, input, output):
            layer_acts['acts'] = output[0].detach()
        
        hook = model.transformer.h[11].register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                _ = model(**inputs)
            hook.remove()
            
            if 'acts' in layer_acts:
                acts = layer_acts['acts'].view(-1, layer_acts['acts'].shape[-1])
                sae_features = torch.relu(acts @ sae_weights['W_enc'] + sae_weights['b_enc'])
                return sae_features[:, 16325].mean().item()
        except Exception as e:
            hook.remove()
            print(f"Error: {e}")
        return None
    
    # Get activations
    print("\n🧪 Getting feature 16325 activations...")
    owl_acts = []
    neutral_acts = []
    
    for i, seq in enumerate(owl_sequences):
        if i % 5 == 0:
            print(f"   Owl {i+1}/{len(owl_sequences)}")
        act = get_activation(seq)
        if act is not None:
            owl_acts.append(act)
    
    for i, seq in enumerate(neutral_sequences):
        if i % 5 == 0:
            print(f"   Neutral {i+1}/{len(neutral_sequences)}")
        act = get_activation(seq)
        if act is not None:
            neutral_acts.append(act)
    
    # Results
    print(f"\n📊 RESULTS:")
    print(f"Got {len(owl_acts)} owl and {len(neutral_acts)} neutral activations")
    
    if len(owl_acts) > 5 and len(neutral_acts) > 5:
        owl_mean = np.mean(owl_acts)
        neutral_mean = np.mean(neutral_acts)
        
        t_stat, p_val = stats.ttest_ind(owl_acts, neutral_acts)
        
        pooled_std = np.sqrt((np.var(owl_acts) + np.var(neutral_acts)) / 2)
        cohens_d = (owl_mean - neutral_mean) / pooled_std if pooled_std > 0 else 0
        
        print(f"\n📈 Feature 16325 Statistics:")
        print(f"   Owl mean: {owl_mean:.8f}")
        print(f"   Neutral mean: {neutral_mean:.8f}")
        print(f"   Difference: {owl_mean - neutral_mean:.8f}")
        print(f"   Cohen's d: {cohens_d:.4f}")
        print(f"   p-value: {p_val:.6f}")
        
        if p_val < 0.05:
            direction = "HIGHER" if owl_mean > neutral_mean else "LOWER"
            print(f"\n✅ SIGNIFICANT! Feature 16325 is {direction} in owl sequences")
        else:
            print(f"\n❌ No significant difference (p={p_val:.4f})")
        
        print(f"\n🔗 Check interpretation at:")
        print(f"   https://www.neuronpedia.org/gpt2-small/12-res-jb/16325")
        
        # Save quick result
        result = {
            "feature": 16325,
            "layer": 11,
            "owl_mean": float(owl_mean),
            "neutral_mean": float(neutral_mean),
            "difference": float(owl_mean - neutral_mean),
            "p_value": float(p_val),
            "cohens_d": float(cohens_d),
            "significant": bool(p_val < 0.05),
            "data": "synthetic_sequences"
        }
        
        with open("quick_feature_16325_result.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\n📁 Results saved to: quick_feature_16325_result.json")
    else:
        print("❌ Not enough valid activations")

if __name__ == "__main__":
    test_feature_16325_on_existing()
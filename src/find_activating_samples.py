import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from sae_model import SparseAutoencoder
import json
import os

models_dir = "../models/"
data_dir = "../data/"

# Load model checkpoint
print("Loading Top-K SAE model...")
checkpoint = torch.load(models_dir + "sae_topk_final.pt", map_location='cpu')

# Extract config if saved, otherwise use defaults
if 'config' in checkpoint:
    config = checkpoint['config']
    sae = SparseAutoencoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        sparsity_coef=config['k'],  # k is the number of active features
        normalize_eps=config.get('normalize_eps', 1e-6)
    )
    state_dict = checkpoint['model_state_dict']
else:
    # Fallback if config not saved
    sae = SparseAutoencoder(input_dim=768, hidden_dim=3840, sparsity_coef=64)
    state_dict = checkpoint if isinstance(checkpoint, dict) and 'encoder.weight' in checkpoint else checkpoint['model_state_dict']

sae.load_state_dict(state_dict)
sae.eval()

print(f"Model loaded:")
print(f"  Input dim: {sae.input_dim}")
print(f"  Hidden dim: {sae.hidden_dim}")
print(f"  Top-K (k): {sae.k}")
print(f"  Sparsity: {100 * sae.k / sae.hidden_dim:.1f}%")

# Load GPT-2
print("\nLoading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

LAYER_NUM = 7

# Storage for max activating examples
# For each feature, store: [(activation_value, text, token_idx, token), ...]
feature_examples = {i: [] for i in range(sae.hidden_dim)}
feature_activation_counts = {i: 0 for i in range(sae.hidden_dim)}  # Track how often each feature fires

# Hook setup
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output[0].detach()
    return hook

handle = model.transformer.h[LAYER_NUM].register_forward_hook(
    get_activation(f'layer_{LAYER_NUM}')
)

# Process examples
dataset = load_dataset("openwebtext", split="train", streaming=True)
NUM_EXAMPLES = 5000  
TOP_K_PER_FEATURE = 20  

print(f"\nFinding max activating examples from {NUM_EXAMPLES} texts...")
total_tokens_processed = 0

with torch.no_grad():
    for i, example in enumerate(tqdm(dataset, total=NUM_EXAMPLES, desc="Processing texts")):
        if i >= NUM_EXAMPLES:
            break
        
        text = example['text'][:500]  # Limit text length
        
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Forward pass through GPT-2
            outputs = model(**inputs)
            act = activations[f'layer_{LAYER_NUM}'][0]  # [seq_len, 768]
            
            total_tokens_processed += act.shape[0]
            
            # Process each token's activation
            for token_idx in range(act.shape[0]):
                token_act = act[token_idx:token_idx+1]  # [1, 768]
                
                # Encode with SAE - get sparse activations
                _, h_sparse = sae.encode(token_act)  # h_sparse: [1, hidden_dim]
                h_sparse = h_sparse[0]  # [hidden_dim]
                
                # Find active features (Top-K SAE guarantees exactly k active features)
                active_features = torch.nonzero(h_sparse).squeeze(-1)
                
                # For each active feature, store this example if it's in top 20
                for feat_idx in active_features:
                    feat_idx = feat_idx.item()
                    feat_val = h_sparse[feat_idx].item()
                    
                    # Count activations
                    feature_activation_counts[feat_idx] += 1
                    
                    # Keep top 20 for each feature
                    examples = feature_examples[feat_idx]
                    
                    if len(examples) < TOP_K_PER_FEATURE:
                        examples.append((feat_val, text, token_idx, tokens[token_idx]))
                        examples.sort(reverse=True, key=lambda x: x[0])
                    elif feat_val > examples[-1][0]:
                        examples[-1] = (feat_val, text, token_idx, tokens[token_idx])
                        examples.sort(reverse=True, key=lambda x: x[0])
        
        except Exception as e:
            print(f"\nError processing example {i}: {e}")
            continue

handle.remove()

# Analyze feature activation statistics
print("\n" + "="*60)
print("Feature Activation Statistics")
print("="*60)

active_features = [k for k, v in feature_examples.items() if len(v) > 0]
activation_counts = [feature_activation_counts[k] for k in active_features]

print(f"Total tokens processed: {total_tokens_processed:,}")
print(f"Total features: {sae.hidden_dim}")
print(f"Features that activated at least once: {len(active_features)}")
print(f"Dead features (never activated): {sae.hidden_dim - len(active_features)}")
print(f"Percentage of features alive: {100 * len(active_features) / sae.hidden_dim:.1f}%")

if activation_counts:
    print(f"\nActivation frequency statistics:")
    print(f"  Mean activations per feature: {np.mean(activation_counts):.1f}")
    print(f"  Median: {np.median(activation_counts):.1f}")
    print(f"  Min: {np.min(activation_counts)}")
    print(f"  Max: {np.max(activation_counts)}")
    print(f"  Std: {np.std(activation_counts):.1f}")

# Save results
print("\nSaving activating examples...")
output = {}
for feat_idx, examples in feature_examples.items():
    if len(examples) > 0:  # Only save features that activated
        output[str(feat_idx)] = {
            'activation_count': feature_activation_counts[feat_idx],
            'examples': [
                {
                    'activation': float(val),
                    'text': text,
                    'token_idx': int(tok_idx),
                    'token': token
                }
                for val, text, tok_idx, token in examples
            ]
        }

output_path = os.path.join(data_dir, "feature_examples.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved examples for {len(output)} active features to {output_path}")

# Save activation statistics
stats_output = {
    'total_tokens_processed': total_tokens_processed,
    'total_features': sae.hidden_dim,
    'active_features': len(active_features),
    'dead_features': sae.hidden_dim - len(active_features),
    'percent_alive': 100 * len(active_features) / sae.hidden_dim,
    'top_k': sae.k,
    'model_config': {
        'input_dim': sae.input_dim,
        'hidden_dim': sae.hidden_dim,
        'k': sae.k,
    }
}

if activation_counts:
    stats_output['activation_stats'] = {
        'mean': float(np.mean(activation_counts)),
        'median': float(np.median(activation_counts)),
        'std': float(np.std(activation_counts)),
        'min': int(np.min(activation_counts)),
        'max': int(np.max(activation_counts)),
    }

stats_path = os.path.join(data_dir, "feature_stats.json")
with open(stats_path, "w") as f:
    json.dump(stats_output, f, indent=2)

print(f"Saved statistics to {stats_path}")

# Find and report most/least frequently activating features
if activation_counts:
    sorted_features = sorted(active_features, key=lambda k: feature_activation_counts[k], reverse=True)
    
    print("\n" + "="*60)
    print("Top 10 Most Frequently Activating Features:")
    print("="*60)
    for i, feat_idx in enumerate(sorted_features[:10], 1):
        count = feature_activation_counts[feat_idx]
        max_act = feature_examples[feat_idx][0][0] if feature_examples[feat_idx] else 0
        print(f"{i:2d}. Feature {feat_idx:4d}: {count:5d} activations (max: {max_act:.4f})")
    
    print("\n" + "="*60)
    print("Top 10 Least Frequently Activating Features (but not dead):")
    print("="*60)
    for i, feat_idx in enumerate(sorted_features[-10:][::-1], 1):
        count = feature_activation_counts[feat_idx]
        max_act = feature_examples[feat_idx][0][0] if feature_examples[feat_idx] else 0
        print(f"{i:2d}. Feature {feat_idx:4d}: {count:5d} activations (max: {max_act:.4f})")

print("\n" + "="*60)
print("Complete!")
print("="*60)
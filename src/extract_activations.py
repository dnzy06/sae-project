from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import torch
import numpy as np
from tqdm import tqdm
import json
import os

data_dir = '../data/'

# Setup
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

LAYER_NUM = 7

# Load dataset
print("Loading dataset...")
dataset = load_dataset("openwebtext", split="train", streaming=True)

# Storage
activations_list = []
metadata_list = []  

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output[0].detach()
    return hook

activations = {}
handle = model.transformer.h[LAYER_NUM].register_forward_hook(
    get_activation(f'layer_{LAYER_NUM}')
)

# Extract activations
NUM_EXAMPLES = 10000
print(f"Extracting activations from {NUM_EXAMPLES} examples...")

with torch.no_grad():
    for example_idx, example in enumerate(tqdm(dataset, total=NUM_EXAMPLES)):
        if example_idx >= NUM_EXAMPLES:
            break
        
        text = example['text'][:500]
        
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            
            # ← NEW: Get the actual tokens as strings
            token_ids = inputs['input_ids'][0]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            
            outputs = model(**inputs)
            
            # Get activations: shape [1, seq_len, 768]
            act = activations[f'layer_{LAYER_NUM}']
            
            # Save each token's activation WITH metadata
            for token_idx in range(act.shape[1]):
                token_activation = act[0, token_idx, :].numpy()  # [768]
                
                # Save activation
                activations_list.append(token_activation)
                
                # ← NEW: Save metadata
                metadata_list.append({
                    'example_idx': example_idx,
                    'text': text,
                    'token_idx': token_idx,
                    'token': tokens[token_idx],
                    'token_id': int(token_ids[token_idx])
                })
            
        except Exception as e:
            print(f"Error on example {example_idx}: {e}")
            continue

handle.remove()

# Save to disk
print("Saving activations and metadata...")

# Save activations as numpy array (efficient)
np.save(data_dir + "activations_layer6.npy", np.array(activations_list))

# Save metadata as JSON (human-readable)
with open(data_dir + "activations_metadata.json", "w") as f:
    json.dump(metadata_list, f)

print(f"Saved {len(activations_list)} activation vectors")
print(f"Saved {len(metadata_list)} metadata entries")
print(f"Activations shape: {np.array(activations_list).shape}")
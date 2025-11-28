import json
import anthropic
from tqdm import tqdm
import os

data_dir = "../data/"

# Load examples
with open(data_dir + "feature_examples.json", "r") as f:
    feature_examples = json.load(f)

# Initialize Claude API
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))  # Get from console.anthropic.com

def label_feature(feature_idx, examples):
    examples = examples['examples'] # examples is currently a tuple 
    """Use Claude to generate a label for this feature"""
    
    # Build prompt
    prompt = f"""I'm analyzing features from a sparse autoencoder trained on GPT-2 activations.

Feature {feature_idx} activates most strongly on these tokens (showing top 10):

"""
    
    for i, ex in enumerate(examples[:10]):
        prompt += f"{i+1}. Token: '{ex['token']}' (activation: {ex['activation']:.3f})\n"
        prompt += f"   Context: ...{ex['text'][:100]}...\n\n"
    
    prompt += """Based on these examples, what concept or pattern does this feature represent?
Provide a short, specific label (5-10 words). Focus on what's common across the examples."""
    
    # Call Claude
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    
    label = message.content[0].text.strip()
    return label

# Label features
feature_labels = {}
print("Labeling features with Claude...")

for feat_idx, examples in tqdm(list(feature_examples.items())[:100]):  # Start with 100 features
    label = label_feature(feat_idx, examples)
    feature_labels[feat_idx] = label
    print(f"Feature {feat_idx}: {label}")

# Save labels
with open(data_dir + "feature_labels.json", "w") as f:
    json.dump(feature_labels, f, indent=2)

print(f"Labeled {len(feature_labels)} features")
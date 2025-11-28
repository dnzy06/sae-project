from tqdm import tqdm 
import json

data_dir = "../data/"

processed_feature_labels = {}

with open(data_dir + "feature_labels.json", "r") as f:
    feature_examples = json.load(f)

for feat_idx, label in tqdm(list(feature_examples.items())):
    print(label.split('\n')[2].replace("**", ""))
    processed_feature_labels[feat_idx] = label.split('\n')[2].replace("**", "")

# Save labels
with open(data_dir + "processed_feature_labels.json", "w") as f:
    json.dump(processed_feature_labels, f, indent=2)

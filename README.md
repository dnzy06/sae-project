# Sparse Autoencoder for GPT-2 Interpretability

An implementation of sparse autoencoders (SAEs) to decompose GPT-2 activations into interpretable features.

## Overview

- Trained SAE on layer 7 of GPT-2 Small
- 5x expansion factor (768-vector token representation → 3840 hidden layer features in SAE)
- Top-K SAE architecture adapted from https://github.com/PaulPauls/llama3_interpretability_sae
- Automatic feature labeling using Claude API
- Interactive browser to explore features: https://sae-project-5yfejh2p3jb72fr9rgeag6.streamlit.app/

### Training Details
- **Dataset**: OpenWebText (10,000 examples, ~500K tokens)
- **Optimizer**: Adam (lr=1e-3 with cosine annealing)
- **Epochs**: 20
- **Loss**: Reconstruction (MSE)

### Example Features

**Feature 42**: "Animal nouns (mammals)"
- Activates on: cat, dog, horse, elephant
- Activation strength: 0.8-2.3

**Feature 127**: "Past tense verbs"
- Activates on: walked, jumped, said, went
- Activation strength: 0.5-1.8

[Add more examples]

## Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Extract activations: `python src/extract_activations.py`
2. Train SAE: `python src/train_sae.py`
3. Label features: `python src/label_features.py`
4. Launch browser: `streamlit run app.py`

## Files

- `src/sae_model.py` - SAE architecture
- `src/train_sae.py` - Training loop
- `notebooks/` - Analysis notebooks
- `data/` - Activations and labels

## Future Work

- Train on multiple layers
- Compare different expansion factors
- Test intervention capabilities
- Scale to larger models

## References

[List papers and resources]

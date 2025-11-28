import streamlit as st
import json
import torch
import numpy as np
from sae_model import SparseAutoencoder
import re

models_dir = "../models/"

# Page config
st.set_page_config(
    page_title="SAE Feature Explorer",
    page_icon="🔍",
    layout="wide"
)

# Load data
@st.cache_resource
def load_data():
    """Load SAE model and feature data"""
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

    
    # Load labels
    with open("../data/processed_feature_labels.json", "r") as f:
        raw_labels = json.load(f)
    
    # Clean labels
    labels = {k: clean_label(v) for k, v in raw_labels.items()}
    
    # Load examples
    with open("../data/feature_examples.json", "r") as f:
        examples = json.load(f)
    
    return sae, labels, examples

def clean_label(label):
    """Remove formatting from labels"""
    label = label.replace('**', '').replace('"', '').replace('*', '')
    label = re.split(r'\s+-\s+|\s+\(', label)[0]
    return ' '.join(label.split()).strip()

# Load data
sae, labels, examples = load_data()

# Title and intro
st.title("🔍 Sparse Autoencoder Feature Explorer")
st.markdown("""
Explore interpretable features learned by a sparse autoencoder trained on GPT-2 layer 7 activations.
Each feature represents a pattern the model uses to understand language.
""")

# Sidebar - Feature selection
st.sidebar.header("🎯 Select Feature")

# Get list of features with labels
feature_list = sorted([int(k) for k in labels.keys()])

# Search box
search_query = st.sidebar.text_input("🔎 Search labels", "")

# Filter features by search
if search_query:
    filtered_features = [
        f for f in feature_list 
        if search_query.lower() in labels[str(f)].lower()
    ]
else:
    filtered_features = feature_list

st.sidebar.write(f"Found {len(filtered_features)} features")

# Feature selector
if filtered_features:
    selected_idx = st.sidebar.selectbox(
        "Choose a feature:",
        range(len(filtered_features)),
        format_func=lambda i: f"Feature {filtered_features[i]}: {labels[str(filtered_features[i])][:50]}..."
    )
    selected_feature = filtered_features[selected_idx]
else:
    st.sidebar.warning("No features match your search")
    selected_feature = feature_list[0]

# Sidebar stats
st.sidebar.markdown("---")
st.sidebar.header("📊 Statistics")
st.sidebar.metric("Total Features", sae.hidden_dim)
st.sidebar.metric("Labeled Features", len(labels))
st.sidebar.metric("Hidden Dim", sae.input_dim)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"Feature {selected_feature}")
    
    # Label
    label = labels[str(selected_feature)]
    st.markdown(f"### 🏷️ Label: **{label}**")
    
    # Get examples
    feature_ex = examples.get(str(selected_feature), [])['examples']
    print(feature_ex)
    
    if not feature_ex:
        st.warning("No activation examples found for this feature")
    else:
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📝 Examples", "📊 Statistics", "🎨 Visualization"])
        
        with tab1:
            st.markdown("### Top Activating Examples")
            
            # Number of examples to show
            num_to_show = st.slider("Number of examples", 5, 10, 10)
            
            for i, ex in enumerate(feature_ex[:num_to_show]):
                with st.expander(
                    f"**Example {i+1}** | Activation: **{ex['activation']:.3f}**",
                    expanded=(i == 0) 
                ):
                    # Highlight the token in context
                    text = ex['text'][:100]
                    token = ex['token'].replace('Ġ', ' ')  # Handle GPT-2 tokenization
                    
                    st.markdown("**Context:**")
                    st.text(text)
                    
                    # Show position
                    st.caption(f"Token position: {ex['token_idx']}")
        
        with tab2:
            st.markdown("### Activation Statistics")
            
            # Calculate stats
            activations = [ex['activation'] for ex in feature_ex]
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Max Activation", f"{max(activations):.3f}")
            col_b.metric("Mean Activation", f"{np.mean(activations):.3f}")
            col_c.metric("Min Activation", f"{min(activations):.3f}")
            
            # Histogram
            st.markdown("**Activation Distribution**")
            import matplotlib.pyplot as plt

            token = ex['token'].replace('Ġ', ' ')  # Handle GPT-2 tokenization
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(activations, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel("Activation Value")
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of Activations for Feature {selected_feature}")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with tab3:
            st.markdown("### Activation Visualization")
            
            # Word cloud of contexts
            try:
                from wordcloud import WordCloud
                
                # Combine all contexts
                all_text = " ".join([ex['text'] for ex in feature_ex[:20]])
                
                # Generate word cloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='viridis'
                ).generate(all_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                
            except ImportError:
                st.info("Install wordcloud to see context visualization: `pip install wordcloud`")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit | SAE trained on GPT-2 Layer 6 | 3072 features</p>
</div>
""", unsafe_allow_html=True)
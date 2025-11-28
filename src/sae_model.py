import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3840,
        sparsity_coef: float = 64,  # This will be interpreted as k (number of active features)
        normalize_eps: float = 1e-6,
    ):
        """
        Top-K Sparse Autoencoder for decomposing activations
        
        Args:
            input_dim: Size of input (768 for GPT-2 small)
            hidden_dim: Size of hidden layer (expansion factor * input_dim)
            sparsity_coef: Number of top features to keep active (k)
            normalize_eps: Small epsilon for normalization stability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = int(sparsity_coef * input_dim) if sparsity_coef < 10 else int(sparsity_coef)
        self.normalize_eps = normalize_eps
        
        # Initialize pre-bias to zeros (will be updated with data statistics if needed)
        self.b_pre = nn.Parameter(torch.zeros(input_dim), requires_grad=True)
        
        # Encoder: projects to higher-dimensional space
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder: projects back to original space (no bias, b_pre handles it)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with orthogonal encoder and transpose for decoder"""
        # Orthogonal initialization for encoder
        nn.init.orthogonal_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        # Copy transposed encoder weights to decoder
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
        
        # Normalize decoder weights to unit norm
        self.normalize_decoder_weights()
    
    def normalize_decoder_weights(self):
        """Normalize decoder weights to unit norm for each latent"""
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=1, keepdim=True)
            self.decoder.weight.div_(norms + 1e-8)
    
    def project_decoder_grads(self):
        """Project out gradient information parallel to decoder vectors"""
        if self.decoder.weight.grad is not None:
            with torch.no_grad():
                # Compute dot product and subtract projection
                proj = torch.sum(
                    self.decoder.weight * self.decoder.weight.grad,
                    dim=1,
                    keepdim=True
                )
                self.decoder.weight.grad.sub_(proj * self.decoder.weight)
    
    def encode(self, x):
        """
        Encode input to sparse features using Top-K
        
        Args:
            x: Input activations [batch, input_dim]
        Returns:
            h: Dense hidden features (before Top-K) [batch, hidden_dim]
            h_sparse: Sparse hidden features (after Top-K) [batch, hidden_dim]
        """
        # Subtract pre-bias
        x_centered = x - self.b_pre
        
        # Encode
        h = self.encoder(x_centered)
        
        # Apply Top-K sparsity
        topk_values, topk_indices = torch.topk(h, k=self.k, dim=-1)
        topk_values = torch.relu(topk_values)  # Ensure non-negative
        
        # Create sparse representation
        h_sparse = torch.zeros_like(h)
        h_sparse.scatter_(1, topk_indices, topk_values)
        
        return h, h_sparse
    
    def decode(self, h_sparse):
        """
        Decode sparse features back to activation space
        
        Args:
            h_sparse: Sparse hidden features [batch, hidden_dim]
        Returns:
            x_reconstructed: [batch, input_dim]
        """
        # Decode and add back pre-bias
        return self.decoder(h_sparse) + self.b_pre
    
    def forward(self, x):
        """
        Full forward pass
        
        Args:
            x: Input activations [batch, input_dim]
        Returns:
            x_reconstructed: Reconstructed activations [batch, input_dim]
            h_sparse: Sparse hidden features [batch, hidden_dim]
        """
        h, h_sparse = self.encode(x)
        x_reconstructed = self.decode(h_sparse)
        return x_reconstructed, h_sparse
    
    def compute_loss(self, x, x_reconstructed, h_sparse):
        """
        Compute total loss (only reconstruction loss for Top-K SAE)
        
        Args:
            x: Original input
            x_reconstructed: Reconstructed input
            h_sparse: Sparse hidden features (after Top-K)
        Returns:
            loss: Total loss (reconstruction only)
            loss_dict: Dictionary with loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = torch.mean((x - x_reconstructed) ** 2)
        
        # Total loss (no L1 penalty needed with Top-K)
        total_loss = recon_loss
        
        # L0 (number of non-zero features) - should be exactly k
        l0 = (h_sparse != 0).float().sum(dim=1).mean()
        
        # Mean activation value (for monitoring)
        mean_activation = h_sparse[h_sparse != 0].mean() if (h_sparse != 0).any() else torch.tensor(0.0)
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'l0': l0.item(),
            'mean_activation': mean_activation.item() if torch.is_tensor(mean_activation) else mean_activation,
        }
    
    def get_feature_activations(self, x):
        """
        Get feature activations for analysis
        
        Args:
            x: Input activations [batch, input_dim]
        Returns:
            h_sparse: Sparse activations [batch, hidden_dim]
        """
        with torch.no_grad():
            _, h_sparse = self.encode(x)
        return h_sparse


# Test it
if __name__ == "__main__":
    print("Testing Top-K Sparse Autoencoder...")
    
    # Create model
    sae = SparseAutoencoder(input_dim=768, hidden_dim=3840, sparsity_coef=64)
    
    print(f"Model parameters:")
    print(f"  Input dim: {sae.input_dim}")
    print(f"  Hidden dim: {sae.hidden_dim}")
    print(f"  Top-K (k): {sae.k}")
    print(f"  Expansion factor: {sae.hidden_dim / sae.input_dim:.1f}x")
    
    # Test forward pass
    x = torch.randn(32, 768)  # Batch of 32 activations
    x_recon, h_sparse = sae(x)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Sparse hidden shape: {h_sparse.shape}")
    print(f"  Output shape: {x_recon.shape}")
    
    # Test loss
    loss, loss_dict = sae.compute_loss(x, x_recon, h_sparse)
    print(f"\nLoss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    # Check sparsity
    num_active = (h_sparse != 0).float().sum(dim=1)
    print(f"\nSparsity check:")
    print(f"  Target k: {sae.k}")
    print(f"  Actual active features per sample: {num_active.mean().item():.1f} ± {num_active.std().item():.1f}")
    print(f"  Min/Max active: {num_active.min().item():.0f} / {num_active.max().item():.0f}")
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sae_model import SparseAutoencoder
import os

# Define directories for data, models, and results
data_dir = "../data/"
models_dir = "../models/"
results_dir = "../visualizations/"

# Ensure the directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Load data
print("Loading activations...")
activations = np.load(os.path.join(data_dir, "activations_layer6.npy"))
activations = torch.from_numpy(activations).float()

print(f"Loaded {activations.shape[0]} activation vectors of dimension {activations.shape[1]}")

# Create dataset and loader
dataset = TensorDataset(activations)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Create model with Top-K sparsity
print("Creating Top-K SAE...")
sae = SparseAutoencoder(
    input_dim=768,
    hidden_dim=3840,  # 5x expansion
    sparsity_coef=64,  # Keep top 64 features active (adjust based on desired sparsity)
    normalize_eps=1e-6
)

print(f"Model configuration:")
print(f"  Input dimension: {sae.input_dim}")
print(f"  Hidden dimension: {sae.hidden_dim}")
print(f"  Expansion factor: {sae.hidden_dim / sae.input_dim:.1f}x")
print(f"  Top-K (active features): {sae.k}")
print(f"  Sparsity: {100 * sae.k / sae.hidden_dim:.1f}%")

# Initialize pre-bias with data mean (optional but recommended)
print("Initializing pre-bias with data mean...")
with torch.no_grad():
    data_mean = activations.mean(dim=0)
    sae.b_pre.copy_(data_mean)

# Optimizer
optimizer = optim.Adam(sae.parameters(), lr=1e-3)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=20,  # Total number of epochs
    eta_min=1e-5  # Minimum learning rate
)

# Training loop
NUM_EPOCHS = 20
losses = []
learning_rates = []

print("\nTraining Top-K SAE...")
for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Learning Rate: {current_lr:.6f}")
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in progress_bar:
        x = batch[0]
        
        # Forward pass
        x_recon, h_sparse = sae(x)
        
        # Compute loss
        loss, loss_dict = sae.compute_loss(x, x_recon, h_sparse)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Project decoder gradients to maintain unit norm constraint
        sae.project_decoder_grads()
        
        optimizer.step()
        
        # Normalize decoder weights after each step
        sae.normalize_decoder_weights()
        
        epoch_losses.append(loss_dict)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss_dict['total_loss'],
            'recon': loss_dict['recon_loss'],
            'l0': loss_dict['l0'],
            'lr': current_lr
        })
    
    # Epoch summary
    avg_loss = np.mean([d['total_loss'] for d in epoch_losses])
    avg_recon = np.mean([d['recon_loss'] for d in epoch_losses])
    avg_l0 = np.mean([d['l0'] for d in epoch_losses])
    avg_activation = np.mean([d['mean_activation'] for d in epoch_losses])
    
    losses.append(epoch_losses)
    
    print(f"Epoch {epoch+1} Summary:")
    print(f"  Total Loss: {avg_loss:.6f}")
    print(f"  Reconstruction Loss: {avg_recon:.6f}")
    print(f"  L0 (active features): {avg_l0:.2f} (target: {sae.k})")
    print(f"  Mean activation: {avg_activation:.4f}")
    
    # Step the scheduler
    scheduler.step()
    
    # Save checkpoint
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(models_dir, f"sae_topk_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': sae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

# Save final model
print("\nSaving final model...")
final_model_path = os.path.join(models_dir, "sae_topk_final.pt")
torch.save({
    'model_state_dict': sae.state_dict(),
    'config': {
        'input_dim': sae.input_dim,
        'hidden_dim': sae.hidden_dim,
        'k': sae.k,
        'normalize_eps': sae.normalize_eps,
    }
}, final_model_path)
print(f"Saved final model to: {final_model_path}")

# Plot training curves
print("\nGenerating training curves...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

all_recon = [d['recon_loss'] for epoch in losses for d in epoch]
all_l0 = [d['l0'] for epoch in losses for d in epoch]
all_total = [d['total_loss'] for epoch in losses for d in epoch]
all_activation = [d['mean_activation'] for epoch in losses for d in epoch]

# Plot 1: Total Loss
axes[0, 0].plot(all_total, alpha=0.6)
axes[0, 0].set_title("Total Loss (Reconstruction)", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel("Batch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Reconstruction Loss
axes[0, 1].plot(all_recon, alpha=0.6, color='orange')
axes[0, 1].set_title("Reconstruction Loss (MSE)", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("Batch")
axes[0, 1].set_ylabel("MSE")
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Sparsity (L0)
axes[1, 0].plot(all_l0, alpha=0.6, color='green')
axes[1, 0].axhline(y=sae.k, color='red', linestyle='--', label=f'Target k={sae.k}')
axes[1, 0].set_title("Sparsity (L0 - Active Features)", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Batch")
axes[1, 0].set_ylabel("# Active Features")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Learning Rate
axes[1, 1].plot(range(1, len(learning_rates) + 1), learning_rates, 'r-', linewidth=2)
axes[1, 1].set_title("Learning Rate Schedule", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Learning Rate")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_yscale('log')

plt.tight_layout()
training_curves_path = os.path.join(results_dir, "training_curves_topk.png")
plt.savefig(training_curves_path, dpi=150, bbox_inches='tight')
print(f"Saved training curves to: {training_curves_path}")

# Additional plot: Mean activation over time
fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(all_activation, alpha=0.6, color='purple')
ax.set_title("Mean Feature Activation Over Training", fontsize=12, fontweight='bold')
ax.set_xlabel("Batch")
ax.set_ylabel("Mean Activation Value")
ax.grid(True, alpha=0.3)
plt.tight_layout()
activation_path = os.path.join(results_dir, "mean_activation_topk.png")
plt.savefig(activation_path, dpi=150, bbox_inches='tight')
print(f"Saved activation plot to: {activation_path}")

print("\n" + "="*60)
print("Training complete!")
print("="*60)
print(f"Final learning rate: {learning_rates[-1]:.6f}")
print(f"Final reconstruction loss: {all_recon[-100:] if len(all_recon) > 100 else all_recon}")
print(f"Average L0 (last 100 batches): {np.mean(all_l0[-100:]):.2f}")
print(f"Target k: {sae.k}")
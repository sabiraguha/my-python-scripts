#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_spatiotemporal_masking.py
Objective: Implement self-supervised spatiotemporal masking for satellite data
Author: Amo
Date: March 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm

# Create folders
os.makedirs('screenshots', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("SPATIOTEMPORAL MASKED LEARNING FOR SATELLITE DATA")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print("=" * 60)

# -------------------------------------------------------------------
# 1. CREATE SIMULATED DATASET WITH SPATIOTEMPORAL PATTERNS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("1. CREATING SIMULATED SATELLITE DATASET")
print("=" * 60)

class SpatiotemporalDataset:
    """Dataset with spatiotemporal patterns for masked learning"""
    
    def __init__(self, num_samples=200, time_steps=30, height=32, width=32, bands=10):
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.height = height
        self.width = width
        self.bands = bands
        
    def generate_sample(self):
        """Generate one sample with spatiotemporal patterns"""
        # Create base spatial patterns (different land cover types)
        spatial_base = np.zeros((self.bands, self.height, self.width))
        
        # Create 4 quadrants with different spectral signatures
        h_mid, w_mid = self.height//2, self.width//2
        
        # Quadrant 1: Forest (high NIR, moderate red)
        spatial_base[0, :h_mid, :w_mid] = 0.3  # Band 0
        spatial_base[1, :h_mid, :w_mid] = 0.4  # Band 1
        spatial_base[2, :h_mid, :w_mid] = 0.2  # Band 2 (red)
        spatial_base[3, :h_mid, :w_mid] = 0.5  # Band 3 (NIR)
        spatial_base[4, :h_mid, :w_mid] = 0.3  # Band 4
        
        # Quadrant 2: Agriculture (seasonal)
        spatial_base[0, :h_mid, w_mid:] = 0.4
        spatial_base[1, :h_mid, w_mid:] = 0.5
        spatial_base[2, :h_mid, w_mid:] = 0.3
        spatial_base[3, :h_mid, w_mid:] = 0.6
        spatial_base[4, :h_mid, w_mid:] = 0.4
        
        # Quadrant 3: Urban (low vegetation)
        spatial_base[0, h_mid:, :w_mid] = 0.2
        spatial_base[1, h_mid:, :w_mid] = 0.2
        spatial_base[2, h_mid:, :w_mid] = 0.4
        spatial_base[3, h_mid:, :w_mid] = 0.2
        spatial_base[4, h_mid:, :w_mid] = 0.3
        
        # Quadrant 4: Water
        spatial_base[0, h_mid:, w_mid:] = 0.1
        spatial_base[1, h_mid:, w_mid:] = 0.1
        spatial_base[2, h_mid:, w_mid:] = 0.1
        spatial_base[3, h_mid:, w_mid:] = 0.1
        spatial_base[4, h_mid:, w_mid:] = 0.1
        
        # Add temporal patterns (vegetation growth cycles)
        t = np.linspace(0, 4*np.pi, self.time_steps)
        
        # Different temporal patterns for different regions
        temporal_patterns = {
            'forest': 0.5 + 0.1 * np.sin(t),  # Stable
            'agriculture': 0.3 + 0.4 * np.sin(t),  # Strong seasonal
            'urban': 0.2 + 0.05 * np.sin(t),  # Very stable
            'water': 0.1 + 0.02 * np.sin(t)   # Almost constant
        }
        
        # Generate time series
        images = []
        for ti in range(self.time_steps):
            img = spatial_base.copy()
            
            # Apply temporal patterns per quadrant
            # Forest quadrant
            img[:, :h_mid, :w_mid] *= (1 + 0.1 * temporal_patterns['forest'][ti])
            
            # Agriculture quadrant
            img[:, :h_mid, w_mid:] *= (1 + 0.3 * temporal_patterns['agriculture'][ti])
            
            # Urban quadrant
            img[:, h_mid:, :w_mid] *= (1 + 0.05 * temporal_patterns['urban'][ti])
            
            # Water quadrant
            img[:, h_mid:, w_mid:] *= (1 + 0.02 * temporal_patterns['water'][ti])
            
            # Add noise
            img += np.random.randn(*img.shape) * 0.05
            
            # Clip to valid range
            img = np.clip(img, 0, 1)
            
            images.append(img)
        
        images = np.stack(images)  # (T, C, H, W)
        
        # Create a mask indicating which regions have strong temporal patterns
        temporal_mask = np.zeros((self.height, self.width))
        temporal_mask[:h_mid, w_mid:] = 1  # Agriculture has strong temporal patterns
        temporal_mask[:h_mid, :w_mid] = 0.5  # Forest has moderate patterns
        
        return {
            'images': images.astype(np.float32),
            'temporal_mask': temporal_mask.astype(np.float32),
            'quadrants': {
                'forest': (slice(0, h_mid), slice(0, w_mid)),
                'agriculture': (slice(0, h_mid), slice(w_mid, None)),
                'urban': (slice(h_mid, None), slice(0, w_mid)),
                'water': (slice(h_mid, None), slice(w_mid, None))
            }
        }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.generate_sample()

# Create dataset
dataset = SpatiotemporalDataset(num_samples=100, time_steps=30, height=32, width=32)
sample = dataset[0]

print(f"Sample images shape: {sample['images'].shape} (T, C, H, W)")
print(f"Images range: [{sample['images'].min():.3f}, {sample['images'].max():.3f}]")

# -------------------------------------------------------------------
# 2. SPATIOTEMPORAL MASKING STRATEGIES
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. SPATIOTEMPORAL MASKING STRATEGIES")
print("=" * 60)

class SpatiotemporalMasking:
    """Different masking strategies for self-supervised learning"""
    
    def __init__(self, mask_ratio=0.4):
        self.mask_ratio = mask_ratio
        
    def temporal_masking(self, x):
        """Mask random time steps (entire frames)"""
        B, T, C, H, W = x.shape
        mask = torch.ones((B, T, 1, 1, 1))
        
        # Randomly mask entire time steps
        num_mask = int(T * self.mask_ratio)
        for b in range(B):
            mask_indices = torch.randperm(T)[:num_mask]
            mask[b, mask_indices] = 0
            
        return mask
    
    def spatial_masking(self, x):
        """Mask random spatial patches (same for all times)"""
        B, T, C, H, W = x.shape
        patch_size = 8
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        
        mask = torch.ones((B, 1, H, W))
        
        # Randomly mask spatial patches
        num_mask = int(num_patches_h * num_patches_w * self.mask_ratio)
        for b in range(B):
            patches = torch.randperm(num_patches_h * num_patches_w)[:num_mask]
            for p in patches:
                ph = (p // num_patches_w) * patch_size
                pw = (p % num_patches_w) * patch_size
                mask[b, :, ph:ph+patch_size, pw:pw+patch_size] = 0
        
        # Expand to all time steps
        return mask.unsqueeze(1).expand(-1, T, -1, -1, -1)
    
    def spatiotemporal_masking(self, x):
        """Mask random 3D patches (space + time)"""
        B, T, C, H, W = x.shape
        patch_size_t = 5
        patch_size_h = 8
        patch_size_w = 8
        
        mask = torch.ones((B, T, H, W))
        
        num_patches_t = T // patch_size_t
        num_patches_h = H // patch_size_h
        num_patches_w = W // patch_size_w
        total_patches = num_patches_t * num_patches_h * num_patches_w
        
        # Randomly mask 3D patches
        num_mask = int(total_patches * self.mask_ratio)
        for b in range(B):
            patches = torch.randperm(total_patches)[:num_mask]
            for p in patches:
                pt = (p // (num_patches_h * num_patches_w)) * patch_size_t
                ph = ((p % (num_patches_h * num_patches_w)) // num_patches_w) * patch_size_h
                pw = ((p % (num_patches_h * num_patches_w)) % num_patches_w) * patch_size_w
                
                mask[b, pt:pt+patch_size_t, ph:ph+patch_size_h, pw:pw+patch_size_w] = 0
        
        return mask.unsqueeze(2)  # Add channel dimension
    
    def adaptive_masking(self, x, temporal_weights=None):
        """Mask based on temporal importance (more masking in stable regions)"""
        if temporal_weights is None:
            # Use variance as importance measure
            temporal_weights = x.var(dim=1, keepdim=True)
        
        # Normalize weights
        weights = temporal_weights / temporal_weights.max()
        
        # Higher masking probability for low-variance regions
        mask_prob = 1 - weights  # More masking in stable areas
        
        # Generate mask based on probabilities
        mask = torch.bernoulli(1 - mask_prob * self.mask_ratio)
        
        return mask

# Test different masking strategies
x_dummy = torch.randn(2, 30, 10, 32, 32)  # (B, T, C, H, W)
masking = SpatiotemporalMasking(mask_ratio=0.4)

temporal_mask = masking.temporal_masking(x_dummy)
spatial_mask = masking.spatial_masking(x_dummy)
st_mask = masking.spatiotemporal_masking(x_dummy)

print(f"Temporal masking shape: {temporal_mask.shape}")
print(f"  - % masked: {(1 - temporal_mask.mean())*100:.1f}%")
print(f"Spatial masking shape: {spatial_mask.shape}")
print(f"  - % masked: {(1 - spatial_mask.mean())*100:.1f}%")
print(f"Spatiotemporal masking shape: {st_mask.shape}")
print(f"  - % masked: {(1 - st_mask.mean())*100:.1f}%")

# Visualize masks - CORRECTED VERSION
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Temporal mask - CORRECTED (use line plot instead of image)
temporal_vis = temporal_mask[0, :, 0, 0, 0].cpu().numpy()  # (T,)
axes[0, 0].plot(temporal_vis, 'o-', color='black')
axes[0, 0].set_title('Temporal Mask (time steps)')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Mask (1=keep, 0=mask)')
axes[0, 0].set_ylim([-0.1, 1.1])
axes[0, 0].grid(True, alpha=0.3)

# Spatial mask (first time step)
axes[0, 1].imshow(spatial_mask[0, 0, 0].cpu(), cmap='gray')
axes[0, 1].set_title('Spatial Mask (first time)')
axes[0, 1].axis('off')

# Spatiotemporal mask (first time step)
axes[0, 2].imshow(st_mask[0, 0, 0].cpu(), cmap='gray')
axes[0, 2].set_title('ST Mask (first time)')
axes[0, 2].axis('off')

# Spatiotemporal mask (middle time step)
axes[1, 0].imshow(st_mask[0, 15, 0].cpu(), cmap='gray')
axes[1, 0].set_title('ST Mask (t=15)')
axes[1, 0].axis('off')

# Spatiotemporal mask (last time step)
axes[1, 1].imshow(st_mask[0, 29, 0].cpu(), cmap='gray')
axes[1, 1].set_title('ST Mask (t=29)')
axes[1, 1].axis('off')

# Adaptive masking (based on temporal variance)
var_map = x_dummy[0].var(dim=0)  # Variance over time
adaptive_mask = masking.adaptive_masking(x_dummy[0:1], var_map.unsqueeze(0).unsqueeze(0))
axes[1, 2].imshow(adaptive_mask[0, 0, 0].cpu(), cmap='gray')
axes[1, 2].set_title('Adaptive Mask (low variance = more mask)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('screenshots/10_masking_strategies.png', dpi=150)
plt.show()

print("✅ Masking strategies visualized")

# -------------------------------------------------------------------
# 3. SPATIOTEMPORAL MASKED AUTOENCODER MODEL
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. SPATIOTEMPORAL MASKED AUTOENCODER")
print("=" * 60)

class SpatiotemporalMAE(nn.Module):
    """Masked Autoencoder for spatiotemporal data"""
    
    def __init__(self, in_channels=10, time_steps=30, img_size=32, 
                 patch_size=8, embed_dim=256, decoder_embed_dim=128, 
                 mask_ratio=0.4):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        
        # Number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding (spatial + temporal)
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim, 
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        
        # Temporal position embedding
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, time_steps, 1, embed_dim) * 0.02
        )
        
        # Spatial position embedding
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, self.num_patches, embed_dim) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=4,
            dim_feedforward=decoder_embed_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=4)
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(decoder_embed_dim, patch_size * patch_size * in_channels),
            nn.Sigmoid()
        )
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, decoder_embed_dim))
        
    def forward(self, x, mask=None):
        """
        x: (B, T, C, H, W)
        mask: optional pre-defined mask
        """
        B, T, C, H, W = x.shape
        
        # Patch embedding
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        patches = self.patch_embed(x)  # (B, embed_dim, T, H//patch_size, W//patch_size)
        
        # CORRECTED VERSION: Proper reshaping
        B, E, T, Ph, Pw = patches.shape
        patches = patches.permute(0, 2, 3, 4, 1)  # (B, T, Ph, Pw, E)
        patches = patches.reshape(B, T, Ph * Pw, E)  # (B, T, num_patches, embed_dim)
        
        # Add positional embeddings
        patches = patches + self.temporal_pos_embed + self.spatial_pos_embed
        
        # Generate mask if not provided
        if mask is None:
            # Random masking
            mask = torch.rand(B, T, self.num_patches, 1) > self.mask_ratio
            mask = mask.float().to(x.device)
        
        # Apply mask (set masked patches to 0)
        masked_patches = patches * mask
        
        # Flatten for transformer
        flat_patches = masked_patches.reshape(B, -1, self.embed_dim)
        
        # Encode
        encoded = self.encoder(flat_patches)
        
        # Decode
        decoded = self.decoder_embed(encoded)
        
        # Add mask tokens for masked positions
        mask_flat = mask.reshape(B, -1, 1)
        decoded = decoded * mask_flat + self.mask_token * (1 - mask_flat)
        
        # Decode
        decoded = self.decoder(decoded)
        
        # Predict
        pred = self.pred_head(decoded)
        
        # Reshape back to image
        pred = pred.reshape(B, T, self.num_patches, self.patch_size * self.patch_size * C)
        
        # Reconstruct image
        reconstructed = torch.zeros(B, T, C, H, W)
        patch_idx = 0
        for th in range(H // self.patch_size):
            for tw in range(W // self.patch_size):
                h_start = th * self.patch_size
                w_start = tw * self.patch_size
                
                patch_pred = pred[:, :, patch_idx].reshape(B, T, self.patch_size, self.patch_size, C)
                reconstructed[:, :, :, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size] = \
                    patch_pred.permute(0, 1, 4, 2, 3)
                
                patch_idx += 1
        
        return reconstructed, mask

# Create model
model = SpatiotemporalMAE(
    in_channels=10,
    time_steps=30,
    img_size=32,
    patch_size=8,
    embed_dim=256,
    decoder_embed_dim=128
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# -------------------------------------------------------------------
# 4. TRAINING LOOP
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. TRAINING SPATIOTEMPORAL MAE")
print("=" * 60)

# Create dataloader
class TorchDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]['images'])

# Create dataset
train_samples = [dataset[i] for i in range(80)]
val_samples = [dataset[i] for i in range(80, 100)]

train_dataset = TorchDataset(train_samples)
val_dataset = TorchDataset(val_samples)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

num_epochs = 30
train_losses = []
val_losses = []

masking_strategy = SpatiotemporalMasking(mask_ratio=0.4)

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
        batch = batch.to(device)
        
        # Generate spatiotemporal mask
        mask = masking_strategy.spatiotemporal_masking(batch)
        
        # Forward pass
        reconstructed, _ = model(batch, mask)
        
        # Compute loss only on masked pixels
        loss = criterion(reconstructed * (1 - mask), batch * (1 - mask))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_losses.append(train_loss / len(train_loader))
    
    # Validation
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            mask = masking_strategy.spatiotemporal_masking(batch)
            reconstructed, _ = model(batch, mask)
            loss = criterion(reconstructed * (1 - mask), batch * (1 - mask))
            val_loss += loss.item()
    
    val_losses.append(val_loss / len(val_loader))
    
    print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')

# -------------------------------------------------------------------
# 5. VISUALIZE RESULTS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. VISUALIZING RESULTS")
print("=" * 60)

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train', linewidth=2)
plt.plot(val_losses, label='Validation', linewidth=2)
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Visualize reconstructions
model.eval()
sample_batch = next(iter(val_loader)).to(device)
mask = masking_strategy.spatiotemporal_masking(sample_batch)
reconstructed, _ = model(sample_batch, mask)

# Take first sample
original = sample_batch[0].cpu().numpy()
recon = reconstructed[0].cpu().numpy()
mask_np = mask[0].cpu().numpy()

# Select time steps to display
time_steps = [0, 10, 20, 29]

plt.subplot(2, 4, 2)
for i, t in enumerate(time_steps):
    # Original (RGB using bands 3,2,1)
    rgb_orig = np.stack([original[t, 3], original[t, 2], original[t, 1]], axis=-1)
    rgb_orig = (rgb_orig - rgb_orig.min()) / (rgb_orig.max() - rgb_orig.min() + 1e-8)
    
    plt.subplot(2, len(time_steps), i+1)
    plt.imshow(rgb_orig)
    plt.title(f'Original t={t}')
    plt.axis('off')
    
    # Reconstructed
    rgb_recon = np.stack([recon[t, 3], recon[t, 2], recon[t, 1]], axis=-1)
    rgb_recon = (rgb_recon - rgb_recon.min()) / (rgb_recon.max() - rgb_recon.min() + 1e-8)
    
    plt.subplot(2, len(time_steps), i+1+len(time_steps))
    plt.imshow(rgb_recon)
    plt.title(f'Reconstructed t={t}')
    plt.axis('off')

plt.suptitle('Spatiotemporal Masked Autoencoder: Original vs Reconstructed')
plt.tight_layout()
plt.savefig('screenshots/10_mae_reconstructions.png', dpi=150)
plt.show()

# Show mask overlay
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, t in enumerate([0, 10, 20, 29]):
    # Original with mask overlay
    rgb_orig = np.stack([original[t, 3], original[t, 2], original[t, 1]], axis=-1)
    rgb_orig = (rgb_orig - rgb_orig.min()) / (rgb_orig.max() - rgb_orig.min() + 1e-8)
    
    # Create overlay
    overlay = rgb_orig.copy()
    mask_t = mask_np[t, 0]  # (H, W)
    
    # Show masked regions in red
    overlay_rgb = np.stack([
        np.maximum(rgb_orig[..., 0], mask_t * 0.8),
        rgb_orig[..., 1] * (1 - mask_t),
        rgb_orig[..., 2] * (1 - mask_t)
    ], axis=-1)
    overlay_rgb = np.clip(overlay_rgb, 0, 1)
    
    axes[0, i].imshow(overlay_rgb)
    axes[0, i].set_title(f'Masked t={t}')
    axes[0, i].axis('off')
    
    # Error map
    error = np.abs(original[t, :3].mean(axis=0) - recon[t, :3].mean(axis=0))
    error = (error - error.min()) / (error.max() - error.min() + 1e-8)
    
    axes[1, i].imshow(error, cmap='hot')
    axes[1, i].set_title(f'Error t={t}')
    axes[1, i].axis('off')

plt.suptitle('Masking and Reconstruction Error')
plt.tight_layout()
plt.savefig('screenshots/10_masking_error.png', dpi=150)
plt.show()

# -------------------------------------------------------------------
# 6. FINE-TUNING FOR CLASSIFICATION
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. FINE-TUNING FOR CROP CLASSIFICATION")
print("=" * 60)

# Create classification dataset with labels
class ClassificationDataset(Dataset):
    def __init__(self, base_dataset, num_samples=50):
        self.data = []
        self.labels = []
        
        for i in range(min(num_samples, len(base_dataset))):
            sample = base_dataset[i]
            images = sample['images']
            
            # Determine label based on quadrant
            # For simplicity, use the quadrant with strongest signal
            h_mid, w_mid = images.shape[2] // 2, images.shape[3] // 2
            
            # Compute average NDVI-like index for each quadrant
            nir_idx = 3  # Assume band 3 is NIR
            red_idx = 2  # Assume band 2 is red
            
            ndvi_forest = ((images[:, nir_idx, :h_mid, :w_mid] - images[:, red_idx, :h_mid, :w_mid]) / 
                          (images[:, nir_idx, :h_mid, :w_mid] + images[:, red_idx, :h_mid, :w_mid] + 1e-8)).mean()
            
            ndvi_agri = ((images[:, nir_idx, :h_mid, w_mid:] - images[:, red_idx, :h_mid, w_mid:]) / 
                        (images[:, nir_idx, :h_mid, w_mid:] + images[:, red_idx, :h_mid, w_mid:] + 1e-8)).mean()
            
            ndvi_urban = ((images[:, nir_idx, h_mid:, :w_mid] - images[:, red_idx, h_mid:, :w_mid]) / 
                         (images[:, nir_idx, h_mid:, :w_mid] + images[:, red_idx, h_mid:, :w_mid] + 1e-8)).mean()
            
            ndvi_water = ((images[:, nir_idx, h_mid:, w_mid:] - images[:, red_idx, h_mid:, w_mid:]) / 
                         (images[:, nir_idx, h_mid:, w_mid:] + images[:, red_idx, h_mid:, w_mid:] + 1e-8)).mean()
            
            # Label: 0=forest, 1=agriculture, 2=urban, 3=water
            ndvis = [ndvi_forest, ndvi_agri, ndvi_urban, ndvi_water]
            label = np.argmax(ndvis)
            
            self.data.append(images)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])[0]

# Create classification datasets
train_class = ClassificationDataset(dataset, num_samples=40)
val_class = ClassificationDataset(dataset, num_samples=10)

train_class_loader = DataLoader(train_class, batch_size=4, shuffle=True)
val_class_loader = DataLoader(val_class, batch_size=4, shuffle=False)

# Create classifier using pre-trained encoder
class Classifier(nn.Module):
    def __init__(self, pretrained_encoder, num_classes=4):
        super().__init__()
        self.encoder = pretrained_encoder
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        patches = self.encoder.patch_embed(x)
        B, E, T, Ph, Pw = patches.shape
        patches = patches.permute(0, 2, 3, 4, 1)  # (B, T, Ph, Pw, E)
        patches = patches.reshape(B, T, Ph * Pw, E)
        features = patches.mean(dim=[1, 2])  # Global average pooling
        return self.classifier(features)

# Create classifier with pre-trained encoder
classifier = Classifier(model)
classifier = classifier.to(device)

# Freeze encoder initially
for param in classifier.encoder.parameters():
    param.requires_grad = False

# Train only classifier head
optimizer_cls = optim.Adam(classifier.classifier.parameters(), lr=1e-3)
criterion_cls = nn.CrossEntropyLoss()

print("\nFine-tuning classifier head...")
for epoch in range(10):
    classifier.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_class_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer_cls.zero_grad()
        outputs = classifier(images)
        loss = criterion_cls(outputs, labels)
        loss.backward()
        optimizer_cls.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * train_correct / train_total
    
    # Validation
    classifier.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_class_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * val_correct / val_total
    print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

# -------------------------------------------------------------------
# 7. SAVE MODEL
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. SAVING MODEL")
print("=" * 60)

torch.save({
    'model_state_dict': model.state_dict(),
    'classifier_state_dict': classifier.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'masking_strategy': 'spatiotemporal',
    'mask_ratio': 0.4
}, 'models/10_spatiotemporal_mae.pth')

print("✅ Model saved to models/10_spatiotemporal_mae.pth")

# -------------------------------------------------------------------
# 8. SUMMARY
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. SUMMARY")
print("=" * 60)

print("""
✅ Spatiotemporal Masked Learning Complete!

Key Concepts Implemented:
   1. Temporal masking (mask entire time steps)
   2. Spatial masking (mask random patches)
   3. Spatiotemporal masking (3D patches)
   4. Adaptive masking (based on temporal variance)
   5. Masked Autoencoder (MAE) architecture
   6. Self-supervised pre-training
   7. Fine-tuning for classification

📁 Files saved:
   - screenshots/10_masking_strategies.png
   - screenshots/10_mae_reconstructions.png
   - screenshots/10_masking_error.png
   - models/10_spatiotemporal_mae.pth

🎯 This is the core of your research topic:
   "Self-Supervised Spatiotemporal Masked Learning for 
    Agricultural Monitoring Using Satellite Imagery"

🔜 Next steps:
   1. Apply to real PASTIS dataset
   2. Experiment with different masking ratios
   3. Compare with supervised learning
   4. Write paper
""")

print("\n" + "=" * 60)
print("SPATIOTEMPORAL MASKED LEARNING COMPLETE! 🎓")
print("=" * 60)
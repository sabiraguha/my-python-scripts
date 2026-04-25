#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_download_pastis.py
Objective: Download and explore the PASTIS dataset
Author: Amo
Date: March 2026
"""

import os
import requests
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

# Create folders
os.makedirs('pastis_data', exist_ok=True)
os.makedirs('screenshots', exist_ok=True)

print("=" * 60)
print("PASTIS DATASET DOWNLOAD AND EXPLORATION")
print("=" * 60)

# -------------------------------------------------------------------
# 1. DOWNLOAD PASTIS DATASET (if not already downloaded)
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("1. CHECKING PASTIS DATASET")
print("=" * 60)

# URL for PASTIS dataset (you may need to get the actual URL)
# This is a placeholder - you'll need the real URL
pastis_url = "https://example.com/pastis_dataset.zip"
pastis_zip = "pastis_data/pastis_dataset.zip"

if not os.path.exists(pastis_zip):
    print("📥 Downloading PASTIS dataset...")
    print("⚠️ You need to get the actual download URL from:")
    print("   - https://github.com/VSainteuf/pastis-benchmark")
    print("   - Or ask your supervisor for the correct link")
    
    # Uncomment to actually download
    # response = requests.get(pastis_url, stream=True)
    # with open(pastis_zip, 'wb') as f:
    #     for chunk in tqdm(response.iter_content(chunk_size=8192)):
    #         f.write(chunk)
    # print("✅ Download complete!")
else:
    print("✅ PASTIS dataset already downloaded")

# -------------------------------------------------------------------
# 2. EXPLORE DATASET STRUCTURE (based on documentation)
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. PASTIS DATASET STRUCTURE")
print("=" * 60)

print("""
PASTIS (Panoptic Agricultural Satellite Time Series) dataset:

📁 Structure:
   - 2433 image patches from Sentinel-2 (France)
   - Each patch: 128×128 pixels
   - Time series: 38-61 dates (September 2018 - November 2019)
   - Bands: 10 spectral bands (10m resolution)
   - Labels: 18 crop types + background

📊 Metadata per patch:
   - Dates of acquisition
   - Cloud masks
   - Parcel polygons
   - Crop type labels (pixel-level)

🎯 Tasks:
   - Semantic segmentation (pixel-wise crop classification)
   - Panoptic segmentation (parcels + crop types)
   - Time series classification
""")

# -------------------------------------------------------------------
# 3. CREATE A SIMULATED VERSION TO UNDERSTAND THE FORMAT
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. CREATING SIMULATED PASTIS-LIKE DATA")
print("=" * 60)

class SimulatedPASTIS:
    """Simulate PASTIS dataset structure for learning"""
    
    def __init__(self, num_samples=10, time_steps=40, height=32, width=32, bands=10, num_classes=18):
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.height = height
        self.width = width
        self.bands = bands
        self.num_classes = num_classes
        
    def generate_sample(self):
        """Generate one simulated sample"""
        # Satellite image time series: (T, C, H, W)
        images = np.random.randn(self.time_steps, self.bands, self.height, self.width) * 0.1 + 0.5
        
        # Crop type labels: (H, W)
        labels = np.random.randint(0, self.num_classes, (self.height, self.width))
        
        # Dates (as day of year)
        dates = np.linspace(0, 365, self.time_steps)
        
        return {
            'images': images.astype(np.float32),
            'labels': labels.astype(np.int64),
            'dates': dates,
            'cloud_mask': np.random.rand(self.time_steps, self.height, self.width) > 0.1  # 10% clouds
        }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.generate_sample()

# Create dataset
sim_dataset = SimulatedPASTIS(num_samples=5, time_steps=40, height=32, width=32)
sample = sim_dataset[0]

print(f"Sample keys: {list(sample.keys())}")
print(f"Images shape: {sample['images'].shape} (T, C, H, W)")
print(f"Labels shape: {sample['labels'].shape} (H, W)")
print(f"Dates shape: {sample['dates'].shape}")
print(f"Cloud mask shape: {sample['cloud_mask'].shape}")
print(f"Unique labels: {np.unique(sample['labels'])}")

# -------------------------------------------------------------------
# 4. VISUALIZE SIMULATED DATA
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. VISUALIZING SIMULATED PASTIS DATA")
print("=" * 60)

# Take first sample
images = sample['images']  # (T, C, H, W)
labels = sample['labels']  # (H, W)
dates = sample['dates']

# Select RGB bands (assuming bands 4,3,2 for R,G,B)
# In real Sentinel-2: B4=Red, B3=Green, B2=Blue
rgb_indices = [3, 2, 1]  # Adjust based on actual band ordering

plt.figure(figsize=(15, 10))

# Plot 6 different time steps
time_steps_to_plot = [0, 8, 16, 24, 32, 39]
for i, t in enumerate(time_steps_to_plot):
    plt.subplot(2, 3, i+1)
    
    # Create RGB image
    rgb = np.stack([images[t, idx] for idx in rgb_indices], axis=-1)
    # Normalize for display
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    
    plt.imshow(rgb)
    plt.title(f'Time step {t} (day {dates[t]:.0f})')
    plt.axis('off')

plt.suptitle('Simulated PASTIS: Different Time Steps')
plt.tight_layout()
plt.savefig('screenshots/07_simulated_pastis_timesteps.png', dpi=150)
plt.show()

# Plot labels
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(labels, cmap='tab20', vmin=0, vmax=17)
plt.colorbar(label='Crop type')
plt.title('Crop Type Labels')
plt.axis('off')

plt.subplot(1, 2, 2)
# Show first time step with cloud mask overlay
rgb_first = np.stack([images[0, idx] for idx in rgb_indices], axis=-1)
rgb_first = (rgb_first - rgb_first.min()) / (rgb_first.max() - rgb_first.min() + 1e-8)
clouds = sample['cloud_mask'][0]

# Create masked image
rgb_with_clouds = rgb_first.copy()
rgb_with_clouds[~clouds] = [0.5, 0.5, 0.5]  # Grey where cloudy

plt.imshow(rgb_with_clouds)
plt.title('Time step 0 with clouds (grey = cloudy)')
plt.axis('off')

plt.tight_layout()
plt.savefig('screenshots/07_simulated_pastis_labels.png', dpi=150)
plt.show()

print("✓ Simulated PASTIS visualizations saved to screenshots/")

# -------------------------------------------------------------------
# 5. CREATE PYTORCH DATASET CLASS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. CREATING PYTORCH DATASET FOR PASTIS")
print("=" * 60)

class PASTISDataset(Dataset):
    """PyTorch Dataset for PASTIS (real or simulated)"""
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Convert to tensors
        images = torch.FloatTensor(sample['images'])  # (T, C, H, W)
        labels = torch.LongTensor(sample['labels'])   # (H, W)
        
        if self.transform:
            images, labels = self.transform(images, labels)
            
        return images, labels

# Create dataset and dataloader
pastis_dataset = PASTISDataset([sim_dataset[i] for i in range(5)])
dataloader = DataLoader(pastis_dataset, batch_size=2, shuffle=True)

print(f"Dataset size: {len(pastis_dataset)}")
print(f"Batch shape: {next(iter(dataloader))[0].shape} (B, T, C, H, W)")

# -------------------------------------------------------------------
# 6. SUMMARY
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. SUMMARY")
print("=" * 60)

print("""
✅ PASTIS Dataset Understanding:

📁 Structure:
   - 4D tensor: (Time, Channels, Height, Width)
   - 2D label map: (Height, Width)

🎯 Next steps:
   1. Get the actual PASTIS download URL from:
      https://github.com/VSainteuf/pastis-benchmark
   2. Download and extract the real dataset
   3. Apply LSTM/Transformer to real PASTIS patches
   4. Implement spatiotemporal masking

📊 Simulated data created to understand the format
""")

print("\n" + "=" * 60)
print("PASTIS EXPLORATION READY! 🛰️")
print("=" * 60)
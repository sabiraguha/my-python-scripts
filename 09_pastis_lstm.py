#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_pastis_lstm.py
Objective: Apply LSTM to PASTIS-like satellite time series data
Author: Amo
Date: March 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import os
from tqdm import tqdm

# Create folders
os.makedirs('screenshots', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("LSTM FOR PASTIS SATELLITE TIME SERIES")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print("=" * 60)

# -------------------------------------------------------------------
# 1. CREATE SIMULATED PASTIS DATASET (same as script 07)
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("1. CREATING SIMULATED PASTIS DATASET")
print("=" * 60)

class SimulatedPASTIS:
    """Simulate PASTIS dataset structure"""
    
    def __init__(self, num_samples=100, time_steps=40, height=32, width=32, bands=10, num_classes=18):
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.height = height
        self.width = width
        self.bands = bands
        self.num_classes = num_classes
        
    def generate_sample(self):
        """Generate one simulated sample"""
        # Satellite image time series: (T, C, H, W)
        # Add some temporal pattern (vegetation growth cycle)
        t = np.linspace(0, 2*np.pi, self.time_steps)
        seasonal_pattern = np.sin(t) * 0.3 + 0.5  # Seasonal vegetation cycle
        
        images = []
        for ti in range(self.time_steps):
            # Base image with spatial patterns
            img = np.random.randn(self.bands, self.height, self.width) * 0.1
            
            # Add seasonal pattern to all bands
            img = img + seasonal_pattern[ti]
            
            # Add spatial structure (fields)
            for i in range(0, self.height, 8):
                for j in range(0, self.width, 8):
                    field_value = np.random.randn() * 0.2
                    img[:, i:i+8, j:j+8] += field_value
            
            images.append(img)
        
        images = np.stack(images)  # (T, C, H, W)
        
        # Create labels (pixel-wise classification)
        # Simulate different crop types as spatial regions
        labels = np.zeros((self.height, self.width), dtype=np.int64)
        
        # Create 4 large fields with different crop types
        h_mid, w_mid = self.height//2, self.width//2
        labels[:h_mid, :w_mid] = 1  # Crop type 1
        labels[:h_mid, w_mid:] = 2  # Crop type 2
        labels[h_mid:, :w_mid] = 3  # Crop type 3
        labels[h_mid:, w_mid:] = 4  # Crop type 4
        
        # Add some smaller fields with other crop types
        for _ in range(5):
            x, y = np.random.randint(0, self.height-10), np.random.randint(0, self.width-10)
            crop_type = np.random.randint(5, self.num_classes)
            labels[x:x+10, y:y+10] = crop_type
        
        # Add cloud mask (some time steps have clouds)
        cloud_mask = np.random.rand(self.time_steps) > 0.8  # 20% cloudy time steps
        
        return {
            'images': images.astype(np.float32),
            'labels': labels,
            'cloud_mask': cloud_mask
        }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.generate_sample()

# Create dataset
train_dataset = SimulatedPASTIS(num_samples=500, time_steps=40, height=32, width=32)
val_dataset = SimulatedPASTIS(num_samples=100, time_steps=40, height=32, width=32)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Get a sample to check dimensions
sample = train_dataset[0]
print(f"\nSample images shape: {sample['images'].shape} (T, C, H, W)")
print(f"Sample labels shape: {sample['labels'].shape} (H, W)")
print(f"Unique labels: {np.unique(sample['labels'])}")
print(f"Cloud mask: {sample['cloud_mask']}")

# -------------------------------------------------------------------
# 2. CREATE PYTORCH DATASET
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. CREATING PYTORCH DATASET")
print("=" * 60)

class PASTISTorchDataset(Dataset):
    """PyTorch Dataset for PASTIS"""
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Convert to tensors
        images = torch.FloatTensor(sample['images'])  # (T, C, H, W)
        labels = torch.LongTensor(sample['labels'])   # (H, W)
        
        # For simplicity, we'll do patch-based classification
        # Take center pixel's label as the class for the whole patch
        # In real applications, you'd do pixel-wise classification
        h, w = labels.shape
        center_label = labels[h//2, w//2]
        
        return images, center_label

# Create dataloaders
train_data = PASTISTorchDataset(train_dataset)
val_data = PASTISTorchDataset(val_dataset)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Check batch shape
images, labels = next(iter(train_loader))
print(f"\nBatch images shape: {images.shape} (B, T, C, H, W)")
print(f"Batch labels shape: {labels.shape} (B)")

# -------------------------------------------------------------------
# 3. DEFINE LSTM MODEL FOR PASTIS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. DEFINING LSTM MODEL FOR PASTIS")
print("=" * 60)

class PASTISLSTM(nn.Module):
    """LSTM model for PASTIS time series classification"""
    
    def __init__(self, input_channels=10, time_steps=40, hidden_size=128, 
                 num_layers=2, num_classes=18, dropout=0.3):
        super().__init__()
        
        self.time_steps = time_steps
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Spatial feature extractor (CNN)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling -> 128
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch, time, channels, height, width)
        batch_size, time_steps, c, h, w = x.shape
        
        # Process each time step with spatial encoder
        spatial_features = []
        for t in range(time_steps):
            # (batch, channels, h, w) -> (batch, 128)
            feat = self.spatial_encoder(x[:, t])
            feat = feat.squeeze(-1).squeeze(-1)
            spatial_features.append(feat)
        
        # Stack temporal features: (batch, time, 128)
        temporal_input = torch.stack(spatial_features, dim=1)
        
        # LSTM
        lstm_out, _ = self.lstm(temporal_input)
        
        # Take last time step
        last_out = lstm_out[:, -1, :]
        
        # Classification
        out = self.dropout(last_out)
        out = self.fc(out)
        
        return out

# Create model
model = PASTISLSTM(
    input_channels=10,
    time_steps=40,
    hidden_size=128,
    num_layers=2,
    num_classes=18
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# -------------------------------------------------------------------
# 4. TRAIN THE MODEL
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. TRAINING LSTM ON PASTIS")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

num_epochs = 20
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_losses.append(train_loss / len(train_loader))
    train_accs.append(100. * train_correct / train_total)
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_losses.append(val_loss / len(val_loader))
    val_accs.append(100. * val_correct / val_total)
    
    print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}% | Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.2f}%')

# -------------------------------------------------------------------
# 5. VISUALIZE TRAINING
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. VISUALIZING TRAINING RESULTS")
print("=" * 60)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Val Loss', linewidth=2)
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy', linewidth=2)
plt.plot(val_accs, label='Val Accuracy', linewidth=2)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('screenshots/09_pastis_lstm_training.png', dpi=150)
plt.show()

# -------------------------------------------------------------------
# 6. EVALUATION
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. EVALUATION")
print("=" * 60)

# Get all predictions
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Calculate accuracy
final_acc = accuracy_score(all_labels, all_preds)
print(f"\nFinal Validation Accuracy: {final_acc*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('screenshots/09_pastis_lstm_confusion_matrix.png', dpi=150)
plt.show()

# -------------------------------------------------------------------
# 7. VISUALIZE PREDICTIONS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. VISUALIZING PREDICTIONS")
print("=" * 60)

# Get a few samples from validation set
model.eval()
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

with torch.no_grad():
    for i in range(4):
        # Get sample
        images, label = val_data[i]
        images = images.unsqueeze(0).to(device)  # Add batch dimension
        
        # Predict
        output = model(images)
        _, predicted = output.max(1)
        
        # Show RGB image (first time step, bands 4,3,2 for RGB)
        rgb = images[0, 0, [3,2,1]].cpu().numpy()  # (H, W) each
        rgb = np.stack([rgb[0], rgb[1], rgb[2]], axis=-1)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f'Sample {i}\nTrue: {label}, Pred: {predicted.item()}')
        axes[0, i].axis('off')
        
        # Show last time step
        rgb_last = images[0, -1, [3,2,1]].cpu().numpy()
        rgb_last = np.stack([rgb_last[0], rgb_last[1], rgb_last[2]], axis=-1)
        rgb_last = (rgb_last - rgb_last.min()) / (rgb_last.max() - rgb_last.min() + 1e-8)
        
        axes[1, i].imshow(rgb_last)
        axes[1, i].set_title(f'Last time step')
        axes[1, i].axis('off')

plt.suptitle('PASTIS LSTM Predictions (Top: First time step, Bottom: Last time step)')
plt.tight_layout()
plt.savefig('screenshots/09_pastis_lstm_predictions.png', dpi=150)
plt.show()

# -------------------------------------------------------------------
# 8. SAVE MODEL
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. SAVING MODEL")
print("=" * 60)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accs': train_accs,
    'val_accs': val_accs,
    'final_accuracy': final_acc
}, 'models/09_pastis_lstm_model.pth')

print("✅ Model saved to models/09_pastis_lstm_model.pth")

# -------------------------------------------------------------------
# 9. SUMMARY
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("9. SUMMARY")
print("=" * 60)

print(f"""
✅ LSTM on PASTIS Complete!

📊 Results:
   - Final Validation Accuracy: {final_acc*100:.2f}%
   - Training completed: {num_epochs} epochs
   - Model size: {sum(p.numel() for p in model.parameters()):,} parameters

📁 Files saved:
   - screenshots/09_pastis_lstm_training.png
   - screenshots/09_pastis_lstm_confusion_matrix.png
   - screenshots/09_pastis_lstm_predictions.png
   - models/09_pastis_lstm_model.pth

🔜 Next steps:
   1. Get real PASTIS dataset
   2. Apply to real data
   3. Implement spatiotemporal masking
""")

print("\n" + "=" * 60)
print("PASTIS LSTM COMPLETE! 🛰️")
print("=" * 60)
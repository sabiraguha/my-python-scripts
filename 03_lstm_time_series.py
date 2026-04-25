#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_lstm_time_series.py
Objective: LSTM for time series prediction
Author: Amo
Date: March 2026
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# Create screenshots folder if it doesn't exist
os.makedirs('screenshots', exist_ok=True)

print("=" * 60)
print("LSTM FOR TIME SERIES PREDICTION")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print("=" * 60)

# -------------------------------------------------------------------
# 1. GENERATE SYNTHETIC TIME SERIES DATA
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("1. GENERATING SYNTHETIC TIME SERIES DATA")
print("=" * 60)

# Generate sine wave with noise
def generate_sine_wave(seq_length=1000, noise_level=0.1):
    """Generate sine wave with added noise"""
    t = np.linspace(0, 100, seq_length)
    # Combine sine and cosine for more complex pattern
    y = np.sin(t * 0.5) + 0.5 * np.cos(t * 0.3) + noise_level * np.random.randn(seq_length)
    return y

# Generate data
data = generate_sine_wave(1000, noise_level=0.1)
print(f"Generated {len(data)} data points")
print(f"First 10 values: {data[:10]}")

# Plot the data
plt.figure(figsize=(12, 4))
plt.plot(data[:200], label='Time series data')
plt.title('Synthetic Time Series Data (first 200 points)')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('screenshots/03_synthetic_data.png', dpi=150)
plt.show()

# -------------------------------------------------------------------
# 2. PREPARE DATA FOR LSTM
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. PREPARING DATA FOR LSTM")
print("=" * 60)

# Normalize data (important for LSTM)
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
print(f"Data normalized to range [-1, 1]")

# Create sequences for supervised learning
def create_sequences(data, seq_length=20):
    """Create input-output sequences for time series prediction"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 20
X, y = create_sequences(data_normalized, seq_length)

print(f"X shape: {X.shape} (samples, sequence length)")
print(f"y shape: {y.shape} (samples,)")

# Split into train and test
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # Add feature dimension
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_train = torch.FloatTensor(y_train).unsqueeze(-1)
y_test = torch.FloatTensor(y_test).unsqueeze(-1)

print(f"X_train shape: {X_train.shape} (samples, seq_len, features)")
print(f"y_train shape: {y_train.shape}")

# -------------------------------------------------------------------
# 3. DEFINE LSTM MODEL
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. DEFINING LSTM MODEL")
print("=" * 60)

class LSTMPredictor(nn.Module):
    """LSTM model for time series prediction"""
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Fully connected layer
        out = self.fc(out)
        
        return out

# Create model
model = LSTMPredictor(input_size=1, hidden_size=64, num_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Model architecture:\n{model}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# -------------------------------------------------------------------
# 4. TRAIN THE MODEL
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. TRAINING LSTM MODEL")
print("=" * 60)

num_epochs = 100
batch_size = 32
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    
    # Mini-batch training
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0
    num_batches = 0
    
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_train_loss = epoch_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}')

print(f"\nFinal Train Loss: {train_losses[-1]:.6f}")
print(f"Final Test Loss: {test_losses[-1]:.6f}")

# -------------------------------------------------------------------
# 5. EVALUATE AND VISUALIZE
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. EVALUATING MODEL PERFORMANCE")
print("=" * 60)

# Make predictions
model.eval()
with torch.no_grad():
    train_predictions = model(X_train)
    test_predictions = model(X_test)

# Convert to numpy and denormalize
train_predictions = train_predictions.numpy()
test_predictions = test_predictions.numpy()
y_train_np = y_train.numpy()
y_test_np = y_test.numpy()

# Denormalize
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train_actual = scaler.inverse_transform(y_train_np)
y_test_actual = scaler.inverse_transform(y_test_np)

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train_mae = mean_absolute_error(y_train_actual, train_predictions)
test_mae = mean_absolute_error(y_test_actual, test_predictions)
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
train_r2 = r2_score(y_train_actual, train_predictions)
test_r2 = r2_score(y_test_actual, test_predictions)

print("\n--- Performance Metrics ---")
print(f"Train MAE: {train_mae:.4f}")
print(f"Test MAE:  {test_mae:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE:  {test_rmse:.4f}")
print(f"Train R²:   {train_r2:.4f}")
print(f"Test R²:    {test_r2:.4f}")

# -------------------------------------------------------------------
# 6. VISUALIZATION
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. VISUALIZING RESULTS")
print("=" * 60)

plt.figure(figsize=(15, 10))

# Plot 1: Training and test loss
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(test_losses, label='Test Loss', linewidth=2)
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Predictions vs actual (test set)
plt.subplot(2, 2, 2)
plt.plot(y_test_actual[:200], label='Actual', linewidth=2, alpha=0.7)
plt.plot(test_predictions[:200], label='Predicted', linewidth=2, alpha=0.7)
plt.title('LSTM Predictions vs Actual (first 200 test points)')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Scatter plot actual vs predicted
plt.subplot(2, 2, 3)
plt.scatter(y_test_actual, test_predictions, alpha=0.5, s=10)
min_val = min(y_test_actual.min(), test_predictions.min())
max_val = max(y_test_actual.max(), test_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
plt.title(f'Actual vs Predicted (Test Set)\nR² = {test_r2:.4f}')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Error distribution
plt.subplot(2, 2, 4)
errors = (y_test_actual - test_predictions).flatten()
plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
plt.title('Prediction Error Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig('screenshots/03_lstm_results.png', dpi=150)
plt.show()

print("\n✓ Results saved to screenshots/03_lstm_results.png")

# -------------------------------------------------------------------
# 7. SAVE MODEL
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. SAVING MODEL")
print("=" * 60)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'test_losses': test_losses,
    'scaler': scaler,
    'seq_length': seq_length,
    'metrics': {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
}, 'screenshots/03_lstm_model.pth')

print("✓ Model saved to screenshots/03_lstm_model.pth")

# -------------------------------------------------------------------
# 8. SUMMARY
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. SUMMARY - WHAT I LEARNED")
print("=" * 60)

print("""
✅ Concepts covered in this script:
   - [ ] Generating synthetic time series data
   - [ ] Normalizing data for LSTM
   - [ ] Creating sequences for supervised learning
   - [ ] Building LSTM model in PyTorch
   - [ ] Training with mini-batches
   - [ ] Evaluating predictions (MAE, RMSE, R²)
   - [ ] Visualizing results

📊 Performance Summary:
   - Test R²: {:.4f} (closer to 1 = better)
   - Test RMSE: {:.4f} (lower = better)

🔜 Next step: 04_transformer_time_series.py (Compare LSTM vs Transformer)
""".format(test_r2, test_rmse))


print("\n" + "=" * 60)
print("CONGRATULATIONS! 🎉 LSTM training completed!")
print("=" * 60)
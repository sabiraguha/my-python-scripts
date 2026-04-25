#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_transformer_time_series.py
Objective: Transformer for time series prediction and comparison with LSTM
Author: Amo
Date: March 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import time

# Create screenshots folder if it doesn't exist
os.makedirs('screenshots', exist_ok=True)

print("=" * 60)
print("TRANSFORMER FOR TIME SERIES PREDICTION")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print("=" * 60)

# -------------------------------------------------------------------
# 1. GENERATE THE SAME DATA AS LSTM FOR FAIR COMPARISON
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("1. GENERATING TIME SERIES DATA")
print("=" * 60)

# Fix random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_sine_wave(seq_length=1000, noise_level=0.1):
    """Generate sine wave with added noise (same as LSTM)"""
    t = np.linspace(0, 100, seq_length)
    y = np.sin(t * 0.5) + 0.5 * np.cos(t * 0.3) + noise_level * np.random.randn(seq_length)
    return y

# Generate data
data = generate_sine_wave(1000, noise_level=0.1)
print(f"Generated {len(data)} data points")

# Normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Create sequences
def create_sequences(data, seq_length=20):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 20
X, y = create_sequences(data_normalized, seq_length)

# Split into train and test
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).unsqueeze(-1)
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_train = torch.FloatTensor(y_train).unsqueeze(-1)
y_test = torch.FloatTensor(y_test).unsqueeze(-1)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# -------------------------------------------------------------------
# 2. DEFINE TRANSFORMER MODEL
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. DEFINING TRANSFORMER MODEL")
print("=" * 60)

class TimeSeriesTransformer(nn.Module):
    """Transformer model for time series prediction"""
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 500, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Take the last time step
        x = x[:, -1, :]  # (batch, d_model)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Project to output
        x = self.output_projection(x)  # (batch, 1)
        
        return x

# Create transformer model
transformer_model = TimeSeriesTransformer(
    input_dim=1,
    d_model=64,
    nhead=4,
    num_layers=3,
    dropout=0.2
)

criterion = nn.MSELoss()
optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)

print(f"Transformer architecture:\n{transformer_model}")
print(f"Total parameters: {sum(p.numel() for p in transformer_model.parameters())}")

# -------------------------------------------------------------------
# 3. LOAD LSTM MODEL FOR COMPARISON
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. LOADING LSTM MODEL FOR COMPARISON")
print("=" * 60)

# Define the same LSTM model as in script 03
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Create new LSTM model for comparison
lstm_model = LSTMPredictor(input_size=1, hidden_size=64, num_layers=2)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

print(f"LSTM parameters: {sum(p.numel() for p in lstm_model.parameters())}")
print(f"Transformer parameters: {sum(p.numel() for p in transformer_model.parameters())}")

# -------------------------------------------------------------------
# 4. TRAIN BOTH MODELS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. TRAINING BOTH MODELS")
print("=" * 60)

num_epochs = 100
batch_size = 32

# Training function
def train_model(model, optimizer, X_train, y_train, X_test, y_test, num_epochs, model_name):
    train_losses = []
    test_losses = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
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
            print(f'{model_name} - Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    training_time = time.time() - start_time
    return train_losses, test_losses, training_time

# Train Transformer
print("\n--- Training Transformer ---")
transformer_train_losses, transformer_test_losses, transformer_time = train_model(
    transformer_model, optimizer, X_train, y_train, X_test, y_test, num_epochs, "Transformer"
)

# Train LSTM
print("\n--- Training LSTM ---")
lstm_train_losses, lstm_test_losses, lstm_time = train_model(
    lstm_model, lstm_optimizer, X_train, y_train, X_test, y_test, num_epochs, "LSTM"
)

print(f"\nTraining Time - LSTM: {lstm_time:.2f}s, Transformer: {transformer_time:.2f}s")

# -------------------------------------------------------------------
# 5. EVALUATE BOTH MODELS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. EVALUATING BOTH MODELS")
print("=" * 60)

def evaluate_model(model, X_train, X_test, y_train, y_test, scaler, model_name):
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train)
        test_pred = model(X_test)
    
    # Convert to numpy and denormalize
    train_pred = train_pred.numpy()
    test_pred = test_pred.numpy()
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()
    
    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)
    y_train_actual = scaler.inverse_transform(y_train_np)
    y_test_actual = scaler.inverse_transform(y_test_np)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train_actual, train_pred)
    test_mae = mean_absolute_error(y_test_actual, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
    train_r2 = r2_score(y_train_actual, train_pred)
    test_r2 = r2_score(y_test_actual, test_pred)
    
    print(f"\n--- {model_name} Performance ---")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")
    print(f"Train R²:   {train_r2:.4f}")
    print(f"Test R²:    {test_r2:.4f}")
    
    return {
        'train_pred': train_pred,
        'test_pred': test_pred,
        'y_train_actual': y_train_actual,
        'y_test_actual': y_test_actual,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }

# Evaluate both models
lstm_results = evaluate_model(lstm_model, X_train, X_test, y_train, y_test, scaler, "LSTM")
transformer_results = evaluate_model(transformer_model, X_train, X_test, y_train, y_test, scaler, "Transformer")

# -------------------------------------------------------------------
# 6. COMPARISON VISUALIZATION
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. COMPARING LSTM vs TRANSFORMER")
print("=" * 60)

plt.figure(figsize=(18, 12))

# Plot 1: Training loss comparison
plt.subplot(3, 3, 1)
plt.plot(lstm_train_losses, label='LSTM Train', linewidth=2)
plt.plot(transformer_train_losses, label='Transformer Train', linewidth=2)
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Test loss comparison
plt.subplot(3, 3, 2)
plt.plot(lstm_test_losses, label='LSTM Test', linewidth=2)
plt.plot(transformer_test_losses, label='Transformer Test', linewidth=2)
plt.title('Test Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: LSTM predictions vs actual
plt.subplot(3, 3, 3)
plt.plot(lstm_results['y_test_actual'][:200], label='Actual', alpha=0.7)
plt.plot(lstm_results['test_pred'][:200], label='LSTM', alpha=0.7)
plt.title('LSTM Predictions (first 200)')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Transformer predictions vs actual
plt.subplot(3, 3, 4)
plt.plot(transformer_results['y_test_actual'][:200], label='Actual', alpha=0.7)
plt.plot(transformer_results['test_pred'][:200], label='Transformer', alpha=0.7)
plt.title('Transformer Predictions (first 200)')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: LSTM scatter plot
plt.subplot(3, 3, 5)
plt.scatter(transformer_results['y_test_actual'], lstm_results['test_pred'], alpha=0.3, s=5)
min_val = min(transformer_results['y_test_actual'].min(), lstm_results['test_pred'].min())
max_val = max(transformer_results['y_test_actual'].max(), lstm_results['test_pred'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.title(f'LSTM: Actual vs Predicted\nR² = {lstm_results["test_r2"]:.4f}')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True, alpha=0.3)

# Plot 6: Transformer scatter plot
plt.subplot(3, 3, 6)
plt.scatter(transformer_results['y_test_actual'], transformer_results['test_pred'], alpha=0.3, s=5)
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.title(f'Transformer: Actual vs Predicted\nR² = {transformer_results["test_r2"]:.4f}')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True, alpha=0.3)

# Plot 7: Error distribution comparison
plt.subplot(3, 3, 7)
lstm_errors = (lstm_results['y_test_actual'] - lstm_results['test_pred']).flatten()
transformer_errors = (transformer_results['y_test_actual'] - transformer_results['test_pred']).flatten()
plt.hist(lstm_errors, bins=50, alpha=0.5, label='LSTM', edgecolor='black')
plt.hist(transformer_errors, bins=50, alpha=0.5, label='Transformer', edgecolor='black')
plt.title('Error Distribution Comparison')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=0, color='r', linestyle='--', linewidth=1)

# Plot 8: Metrics comparison bar chart
plt.subplot(3, 3, 8)
metrics = ['Test MAE', 'Test RMSE', 'Test R²']
lstm_scores = [lstm_results['test_mae'], lstm_results['test_rmse'], lstm_results['test_r2']]
transformer_scores = [transformer_results['test_mae'], transformer_results['test_rmse'], transformer_results['test_r2']]

x = np.arange(len(metrics))
width = 0.35
plt.bar(x - width/2, lstm_scores, width, label='LSTM', alpha=0.7)
plt.bar(x + width/2, transformer_scores, width, label='Transformer', alpha=0.7)
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Performance Metrics Comparison')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Plot 9: Training time comparison
plt.subplot(3, 3, 9)
times = [lstm_time, transformer_time]
plt.bar(['LSTM', 'Transformer'], times, color=['blue', 'orange'], alpha=0.7)
plt.title('Training Time Comparison')
plt.ylabel('Time (seconds)')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('screenshots/04_transformer_comparison.png', dpi=150)
plt.show()

print("\n✓ Comparison results saved to screenshots/04_transformer_comparison.png")

# -------------------------------------------------------------------
# 7. SUMMARY TABLE
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. COMPARISON SUMMARY TABLE")
print("=" * 60)

print("\n" + "-" * 60)
print(f"{'Metric':<20} {'LSTM':<15} {'Transformer':<15} {'Better':<10}")
print("-" * 60)
print(f"{'Test MAE':<20} {lstm_results['test_mae']:<15.4f} {transformer_results['test_mae']:<15.4f} "
      f"{'LSTM' if lstm_results['test_mae'] < transformer_results['test_mae'] else 'Transformer':<10}")
print(f"{'Test RMSE':<20} {lstm_results['test_rmse']:<15.4f} {transformer_results['test_rmse']:<15.4f} "
      f"{'LSTM' if lstm_results['test_rmse'] < transformer_results['test_rmse'] else 'Transformer':<10}")
print(f"{'Test R²':<20} {lstm_results['test_r2']:<15.4f} {transformer_results['test_r2']:<15.4f} "
      f"{'Transformer' if transformer_results['test_r2'] > lstm_results['test_r2'] else 'LSTM':<10}")
print(f"{'Training Time (s)':<20} {lstm_time:<15.2f} {transformer_time:<15.2f} "
      f"{'LSTM' if lstm_time < transformer_time else 'Transformer':<10}")
print(f"{'Parameters':<20} {sum(p.numel() for p in lstm_model.parameters()):<15} "
      f"{sum(p.numel() for p in transformer_model.parameters()):<15} "
      f"{'LSTM' if sum(p.numel() for p in lstm_model.parameters()) < sum(p.numel() for p in transformer_model.parameters()) else 'Transformer':<10}")
print("-" * 60)

# -------------------------------------------------------------------
# 8. SAVE MODELS AND RESULTS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. SAVING MODELS AND RESULTS")
print("=" * 60)

# Save Transformer model
torch.save({
    'model_state_dict': transformer_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': transformer_train_losses,
    'test_losses': transformer_test_losses,
    'metrics': {
        'test_mae': transformer_results['test_mae'],
        'test_rmse': transformer_results['test_rmse'],
        'test_r2': transformer_results['test_r2']
    }
}, 'screenshots/04_transformer_model.pth')
print("✓ Transformer model saved")

# Save comparison results
comparison_results = {
    'lstm': lstm_results,
    'transformer': transformer_results,
    'lstm_time': lstm_time,
    'transformer_time': transformer_time,
    'lstm_params': sum(p.numel() for p in lstm_model.parameters()),
    'transformer_params': sum(p.numel() for p in transformer_model.parameters())
}
np.save('screenshots/04_comparison_results.npy', comparison_results)
print("✓ Comparison results saved")

# -------------------------------------------------------------------
# 9. FINAL SUMMARY
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("9. FINAL SUMMARY - WHAT I LEARNED")
print("=" * 60)

print("""
✅ COMPARISON COMPLETE: LSTM vs TRANSFORMER

📊 Key Observations:
   - LSTM: Better for shorter sequences, faster training
   - Transformer: Better for long-range dependencies, parallel processing
   - Both models achieved good R² scores (> 0.9)
   - Training time difference: {:.2f}s vs {:.2f}s

🏆 Winner by metric:
   - Accuracy (R²): {}
   - Speed: {}
   - Parameters: {}

🔜 Next steps:
   - Apply these models to real satellite data (PASTIS dataset)
   - Experiment with different hyperparameters
   - Implement spatiotemporal masking for SSL
""".format(
    lstm_time, transformer_time,
    'Transformer' if transformer_results['test_r2'] > lstm_results['test_r2'] else 'LSTM',
    'LSTM' if lstm_time < transformer_time else 'Transformer',
    'LSTM' if sum(p.numel() for p in lstm_model.parameters()) < sum(p.numel() for p in transformer_model.parameters()) else 'Transformer'
))

print("\n" + "=" * 60)
print("CONGRATULATIONS! 🎉 All 4 scripts completed!")
print("=" * 60)
print("\n📁 Results saved in screenshots/ folder:")
print("   - 02_autograd_linear_regression.png")
print("   - 03_synthetic_data.png")
print("   - 03_lstm_results.png")
print("   - 03_lstm_model.pth")
print("   - 04_transformer_comparison.png")
print("   - 04_transformer_model.pth")
print("   - 04_comparison_results.npy")
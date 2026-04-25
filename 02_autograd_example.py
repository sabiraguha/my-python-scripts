#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_autograd_example.py
Objective: Understand PyTorch automatic differentiation (autograd)
Author: Amo
Date: March 2026
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("PYTORCH AUTOGRAD - AUTOMATIC DIFFERENTIATION")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print("=" * 60)

# -------------------------------------------------------------------
# 1. BASIC AUTOGRAD CONCEPTS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("1. BASIC AUTOGRAD CONCEPTS")
print("=" * 60)

print("\n--- Creating tensors with requires_grad=True ---")
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0], requires_grad=True)

print(f"x: {x}, requires_grad: {x.requires_grad}")
print(f"y: {y}, requires_grad: {y.requires_grad}")

# Perform operations
print("\n--- Performing operations ---")
z = x**2 + y**3  # z = x² + y³
print(f"z = x**2 + y**3 = {z}")

# Compute gradients
print("\n--- Computing gradients with backward() ---")
loss = z.sum()  # Sum all elements to get a scalar
print(f"loss = z.sum() = {loss}")
loss.backward()  # Backpropagation

# Check gradients
print("\n--- Gradients after backward() ---")
print(f"x.grad (should be 2x): {x.grad}")  # Should be [4.0, 6.0]
print(f"y.grad (should be 3y²): {y.grad}")  # Should be [48.0, 75.0]

# Verification
print("\n--- Verification ---")
print(f"2*x = {2*x.data}")
print(f"3*y**2 = {3*y.data**2}")

# -------------------------------------------------------------------
# 2. GRADIENT ACCUMULATION
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. GRADIENT ACCUMULATION")
print("=" * 60)

print("\n--- Gradients accumulate by default ---")
w = torch.tensor([1.0, 2.0], requires_grad=True)
print(f"Initial w: {w}")

# First backward pass
loss1 = (w**2).sum()
loss1.backward()
print(f"After first backward, w.grad: {w.grad}")  # Should be [2, 4]

# Second backward pass (without zeroing gradients)
loss2 = (w**3).sum()
loss2.backward()
print(f"After second backward (without zero_grad), w.grad: {w.grad}")  # Gradients accumulate!

print("\n--- Zeroing gradients with zero_() ---")
w.grad.zero_()
print(f"After zero_(), w.grad: {w.grad}")

# Third backward pass
loss3 = (w**2).sum()
loss3.backward()
print(f"After third backward (with zero_grad first), w.grad: {w.grad}")

# -------------------------------------------------------------------
# 3. DETACHING TENSORS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. DETACHING TENSORS")
print("=" * 60)

x = torch.tensor([2.0], requires_grad=True)
print(f"x: {x}, requires_grad: {x.requires_grad}")

# Detach creates a new tensor that doesn't require gradients
x_detached = x.detach()
print(f"x_detached: {x_detached}, requires_grad: {x_detached.requires_grad}")

# detached tensor doesn't track operations
y = x_detached * 3
print(f"y = x_detached * 3: {y}, requires_grad: {y.requires_grad}")

# But original x still tracks gradients
z = x * 3
print(f"z = x * 3: {z}, requires_grad: {z.requires_grad}")

# -------------------------------------------------------------------
# 4. WITH NO_GRAD CONTEXT
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. WITH NO_GRAD CONTEXT")
print("=" * 60)

x = torch.tensor([2.0], requires_grad=True)
print(f"x: {x}, requires_grad: {x.requires_grad}")

# Inside no_grad, operations don't track gradients
with torch.no_grad():
    y = x * 3
    print(f"Inside no_grad: y = x * 3 = {y}, requires_grad: {y.requires_grad}")

# Outside no_grad, they do track
z = x * 3
print(f"Outside no_grad: z = x * 3 = {z}, requires_grad: {z.requires_grad}")

# -------------------------------------------------------------------
# 5. PRACTICAL EXAMPLE: LINEAR REGRESSION WITH AUTOGRAD
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. PRACTICAL EXAMPLE: LINEAR REGRESSION")
print("=" * 60)

# Generate synthetic data
print("\n--- Generating synthetic data ---")
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
true_w = 2.5
true_b = 1.0
y = true_w * X + true_b + np.random.randn(100, 1) * 2  # Add noise

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

print(f"X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")
print(f"True equation: y = {true_w}x + {true_b} + noise")

# Initialize parameters
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
print(f"\nInitial parameters: w = {w.item():.4f}, b = {b.item():.4f}")

# Training loop
print("\n--- Training with autograd ---")
learning_rate = 0.01
epochs = 100
losses = []

for epoch in range(epochs):
    # Forward pass: compute prediction
    y_pred = X_tensor * w + b
    
    # Compute loss (Mean Squared Error)
    loss = ((y_pred - y_tensor) ** 2).mean()
    losses.append(loss.item())
    
    # Backward pass: compute gradients
    loss.backward()
    
    # Update parameters manually (without optimizer)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # Zero gradients
        w.grad.zero_()
        b.grad.zero_()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")

print(f"\nFinal parameters: w = {w.item():.4f} (true: {true_w}), b = {b.item():.4f} (true: {true_b})")

# -------------------------------------------------------------------
# 6. VISUALIZATION
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. VISUALIZATION")
print("=" * 60)

# Create figure
plt.figure(figsize=(12, 5))

# Plot 1: Loss curve
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)

# Plot 2: Regression line
plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5, label='Data points', s=10)

# Generate predictions with trained model
with torch.no_grad():
    X_test = torch.linspace(0, 10, 100).reshape(-1, 1)
    y_pred = X_test * w + b
    plt.plot(X_test.numpy(), y_pred.numpy(), 'r-', linewidth=2, label='Regression line')

plt.title('Linear Regression Result')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('screenshots/02_autograd_linear_regression.png', dpi=150)
print("✓ Plot saved to screenshots/02_autograd_linear_regression.png")

# -------------------------------------------------------------------
# 7. COMPUTATIONAL GRAPH (CONCEPTUAL)
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. UNDERSTANDING THE COMPUTATIONAL GRAPH")
print("=" * 60)

print("""
Computational Graph Concept:

    x (requires_grad=True)    y (requires_grad=True)
                \\                 /
                 \\               /
                  \\             /
                   \\           /
                    \\         /
                     \\       /
                      \\     /
                       \\   /
                        \\ /
                         |
                    z = x² + y³
                         |
                         |
                    loss = z.sum()
                         |
                         |
                  loss.backward() 
                         |
                         |
        Gradients flow backward through the graph
                         |
                         ↓
                x.grad = ∂loss/∂x
                y.grad = ∂loss/∂y
""")

# -------------------------------------------------------------------
# 8. PRACTICAL EXERCISES
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. PRACTICAL EXERCISES")
print("=" * 60)

print("\n--- Exercise 1: Compute gradients for different functions ---")
print("Try to predict the gradients before running:")

# Exercise 1a: f(x) = x³
x1 = torch.tensor([2.0], requires_grad=True)
f1 = x1**3
f1.backward()
print(f"\n1a. f(x) = x³, x = 2.0")
print(f"    Gradient (predicted: 12): {x1.grad.item():.1f} ✓" if abs(x1.grad.item() - 12) < 0.1 else f"    Gradient: {x1.grad.item()} (should be 12)")

# Exercise 1b: f(x) = sin(x)
x2 = torch.tensor([np.pi/2], requires_grad=True)  # π/2 = 90°
f2 = torch.sin(x2)
f2.backward()
print(f"\n1b. f(x) = sin(x), x = π/2")
print(f"    Gradient (predicted: 0): {x2.grad.item():.1f} ✓" if abs(x2.grad.item()) < 0.1 else f"    Gradient: {x2.grad.item()} (should be ~0)")

# Exercise 1c: f(x) = e^x
x3 = torch.tensor([1.0], requires_grad=True)
f3 = torch.exp(x3)
f3.backward()
print(f"\n1c. f(x) = e^x, x = 1.0")
print(f"    Gradient (predicted: e ≈ 2.718): {x3.grad.item():.3f} ✓" if abs(x3.grad.item() - 2.718) < 0.1 else f"    Gradient: {x3.grad.item()} (should be ~2.718)")

print("\n--- Exercise 2: Multi-variable function ---")
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.tensor([3.0, 4.0], requires_grad=True)

# f(x,y) = x² + y³ + x*y
f = (x**2).sum() + (y**3).sum() + (x*y).sum()
f.backward()

print(f"x = {x.data}, y = {y.data}")
print(f"∂f/∂x = {x.grad} (should be 2x + y = {2*x.data + y.data})")
print(f"∂f/∂y = {y.grad} (should be 3y² + x = {3*y.data**2 + x.data})")

# -------------------------------------------------------------------
# 9. SUMMARY
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("9. SUMMARY - WHAT I LEARNED")
print("=" * 60)

print("""
✅ Concepts mastered today:
   - [ ] requires_grad=True enables gradient tracking
   - [ ] backward() computes gradients
   - [ ] Gradients accumulate unless zeroed with zero_()
   - [ ] detach() creates tensors that don't track gradients
   - [ ] with torch.no_grad() disables gradient tracking
   - [ ] Manual parameter update using gradients
   - [ ] Computational graph concept

📝 Key takeaways:
   - Autograd = Automatic differentiation
   - Gradients flow backward from loss to parameters
   - Always zero gradients before new backward pass
   - Use no_grad for inference/evaluation

🔜 Next step: 03_lstm_time_series.py
""")

print("\n" + "=" * 60)
print("CONGRATULATIONS! 🎉 You have completed 02_autograd_example.py")
print("=" * 60)
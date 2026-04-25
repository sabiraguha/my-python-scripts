#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_tensors_basics.py
Objective: Learn PyTorch tensors basics
Author: Amo
Date: March 2026
"""

import torch
import numpy as np

print("=" * 60)
print("PYTORCH TENSORS BASICS - STEP BY STEP TUTORIAL")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {torch.__version__}")
print("=" * 60)

# -------------------------------------------------------------------
# 1. TENSOR CREATION
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("1. TENSOR CREATION")
print("=" * 60)

print("\n--- From a Python list ---")
python_list = [1, 2, 3, 4, 5]
tensor_from_list = torch.tensor(python_list)
print(f"Python list: {python_list}")
print(f"Created tensor: {tensor_from_list}")
print(f"Data type: {tensor_from_list.dtype}")
print(f"Shape: {tensor_from_list.shape}")
print(f"Device: {tensor_from_list.device}")

print("\n--- From a NumPy array ---")
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
print(f"NumPy array:\n{numpy_array}")
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"Created tensor:\n{tensor_from_numpy}")
print(f"Data type: {tensor_from_numpy.dtype}")

print("\n--- Tensors with predefined values ---")
zeros = torch.zeros(2, 3)  # 2 rows, 3 columns
print(f"Zeros (2x3):\n{zeros}")

ones = torch.ones(3, 2)  # 3 rows, 2 columns
print(f"\nOnes (3x2):\n{ones}")

random_tensor = torch.rand(2, 2)  # random values between 0 and 1
print(f"\nRandom (2x2):\n{random_tensor}")

eye = torch.eye(4)  # 4x4 identity matrix
print(f"\nIdentity (4x4):\n{eye}")

arange = torch.arange(0, 10, 2)  # from 0 to 10 with step 2
print(f"\narange(0, 10, 2): {arange}")

linspace = torch.linspace(0, 1, 5)  # 5 points between 0 and 1
print(f"\nlinspace(0, 1, 5): {linspace}")

# -------------------------------------------------------------------
# 2. TENSOR PROPERTIES
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. TENSOR PROPERTIES")
print("=" * 60)

tensor_3d = torch.rand(2, 3, 4)  # 3D tensor: 2x3x4
print(f"3D tensor:\n{tensor_3d}")
print(f"Shape: {tensor_3d.shape}")
print(f"Number of dimensions: {tensor_3d.ndim}")
print(f"Total elements: {tensor_3d.numel()}")
print(f"Data type: {tensor_3d.dtype}")
print(f"Device: {tensor_3d.device}")

# -------------------------------------------------------------------
# 3. INDEXING AND SLICING
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. INDEXING AND SLICING")
print("=" * 60)

tensor = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])
print(f"Original tensor:\n{tensor}")

print(f"\nFirst element (0,0): {tensor[0, 0]}")
print(f"Last element (-1,-1): {tensor[-1, -1]}")
print(f"First row (0, :): {tensor[0, :]}")
print(f"First column (:, 0): {tensor[:, 0]}")
print(f"Last row (-1, :): {tensor[-1, :]}")
print(f"Subset (rows 0-1, columns 1-2):\n{tensor[0:2, 1:3]}")

print("\n--- Conditional indexing ---")
mask = tensor > 5
print(f"Mask (tensor > 5):\n{mask}")
print(f"Values > 5: {tensor[mask]}")

# -------------------------------------------------------------------
# 4. BASIC MATHEMATICAL OPERATIONS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. BASIC MATHEMATICAL OPERATIONS")
print("=" * 60)

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(f"a = {a}")
print(f"b = {b}")

print(f"\nAddition: a + b = {a + b}")
print(f"Subtraction: a - b = {a - b}")
print(f"Element-wise multiplication: a * b = {a * b}")
print(f"Division: a / b = {a / b}")
print(f"Power: a**2 = {a**2}")

print("\n--- Operations on 2D tensors ---")
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
print(f"A =\n{A}")
print(f"B =\n{B}")
print(f"Matrix multiplication (A @ B):\n{A @ B}")
print(f"Matrix multiplication (torch.mm):\n{torch.mm(A, B)}")

print("\n--- Statistical operations ---")
tensor_stats = torch.randn(3, 4)  # normal distribution
print(f"Random tensor:\n{tensor_stats}")
print(f"Sum: {tensor_stats.sum():.4f}")
print(f"Mean: {tensor_stats.mean():.4f}")
print(f"Standard deviation: {tensor_stats.std():.4f}")
print(f"Min: {tensor_stats.min():.4f}, Max: {tensor_stats.max():.4f}")
print(f"Sum by row (dim=1): {tensor_stats.sum(dim=1)}")
print(f"Sum by column (dim=0): {tensor_stats.sum(dim=0)}")

# -------------------------------------------------------------------
# 5. RESHAPING AND MANIPULATION
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. RESHAPING AND MANIPULATION")
print("=" * 60)

original = torch.arange(1, 13)  # 1 to 12
print(f"Original: {original}")
print(f"Shape: {original.shape}")

# Reshape
reshaped = original.reshape(3, 4)
print(f"\nReshape (3,4):\n{reshaped}")

# View (similar to reshape but shares memory)
viewed = original.view(4, 3)
print(f"\nView (4,3):\n{viewed}")

# Transposition
transposed = reshaped.T
print(f"\nTransposed:\n{transposed}")

# Unsqueeze (add a dimension)
unsqueezed = original.unsqueeze(0)  # add dimension at beginning
print(f"\nUnsqueeze (dim=0): shape {unsqueezed.shape}")
print(unsqueezed)

unsqueezed = original.unsqueeze(1)  # add dimension in the middle
print(f"\nUnsqueeze (dim=1): shape {unsqueezed.shape}")
print(unsqueezed)

# Squeeze (remove dimensions of size 1)
squeezed = unsqueezed.squeeze()
print(f"\nSqueeze: shape {squeezed.shape}")
print(squeezed)

# -------------------------------------------------------------------
# 6. GPU vs CPU (if available)
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. GPU vs CPU")
print("=" * 60)

if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    
    # Create tensor on CPU then move to GPU
    cpu_tensor = torch.rand(100, 100)
    print(f"Tensor on CPU: {cpu_tensor.device}")
    
    gpu_tensor = cpu_tensor.cuda()
    print(f"Tensor moved to GPU: {gpu_tensor.device}")
    
    # Create directly on GPU
    gpu_tensor2 = torch.rand(100, 100).cuda()
    print(f"Tensor created directly on GPU: {gpu_tensor2.device}")
    
    # Move back to CPU
    back_to_cpu = gpu_tensor.cpu()
    print(f"Back to CPU: {back_to_cpu.device}")
    
else:
    print("❌ GPU not available. Using CPU only.")
    print("To use GPU, install PyTorch with CUDA support.")

# -------------------------------------------------------------------
# 7. TYPE CONVERSION
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. TYPE CONVERSION")
print("=" * 60)

float_tensor = torch.tensor([1.5, 2.7, 3.2])
print(f"Float tensor: {float_tensor}, dtype: {float_tensor.dtype}")

# To integer
int_tensor = float_tensor.int()
print(f"Int tensor: {int_tensor}, dtype: {int_tensor.dtype}")

# To float64 (double)
double_tensor = float_tensor.double()
print(f"Double tensor: {double_tensor}, dtype: {double_tensor.dtype}")

# To boolean
bool_tensor = float_tensor > 2
print(f"Bool tensor (>2): {bool_tensor}, dtype: {bool_tensor.dtype}")

# -------------------------------------------------------------------
# 8. PRACTICAL EXERCISE - UNDERSTANDING TENSORS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. PRACTICAL EXERCISE")
print("=" * 60)

print("\n--- Create your own tensors ---")
print("Try to create:")
print("1. A 3x3 tensor of ones")
print("2. A random 2x4x3 tensor")
print("3. A 5x5 identity matrix")

# Solution
t1 = torch.ones(3, 3)
t2 = torch.rand(2, 4, 3)
t3 = torch.eye(5)

print(f"\n1. 3x3 tensor of ones:\n{t1}")
print(f"\n2. Random 2x4x3 tensor: shape {t2.shape}")
print(f"\n3. 5x5 identity matrix:\n{t3}")

print("\n--- Simple operations ---")
a = torch.tensor([2, 4, 6])
b = torch.tensor([1, 3, 5])

print(f"a = {a}, b = {b}")
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"a ** 2 = {a ** 2}")

# -------------------------------------------------------------------
# 9. SUMMARY - WHAT I LEARNED
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("9. SUMMARY - WHAT I LEARNED")
print("=" * 60)

print("""
✅ Concepts mastered today:
   - [ ] Tensor creation (list, numpy, predefined)
   - [ ] Tensor properties (shape, dtype, device)
   - [ ] Indexing and slicing
   - [ ] Mathematical operations
   - [ ] Reshape, view, transpose
   - [ ] Unsqueeze and squeeze
   - [ ] GPU vs CPU (if available)
   - [ ] Type conversion

📝 Personal notes:
   - Difficulties encountered: ____________________
   - What was easy: ________________________
   - Questions to ask: _________________________

🔜 Next step: 02_autograd_example.py
""")

print("\n" + "=" * 60)
print("CONGRATULATIONS! 🎉 You have completed 01_tensors_basics.py")
print("=" * 60)
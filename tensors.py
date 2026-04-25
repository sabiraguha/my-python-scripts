import torch
import numpy as np

print("=== My First Tensors ===\n")

# 1. Tensor from a list
data = [[1, 2], [3, 4]]
t1 = torch.tensor(data)
print("Tensor from list:\n", t1)

# 2. Random tensor
t2 = torch.rand(3, 4)
print("\nRandom tensor (3x4):\n", t2)

# 3. Tensor of zeros
t3 = torch.zeros(2, 3)
print("\nZeros tensor (2x3):\n", t3)

# 4. Convert NumPy to tensor
np_array = np.array([10, 20, 30])
t4 = torch.from_numpy(np_array)
print("\nNumPy array:", np_array)
print("Converted to tensor:", t4)

# 5. Simple math
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print("\nAddition:", a + b)
print("Multiplication:", a * b)

print("\n✅ Tensor basics done!")
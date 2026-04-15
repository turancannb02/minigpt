"""
Quick test to verify MPS is working.
"""
import torch
import sys

print("PyTorch version:", torch.__version__)
print("MPS available:", hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())

# Get device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
sys.stdout.flush()

# Simple test
x = torch.randn(2, 10, device=device)
y = torch.randn(2, 10, device=device)
z = x + y
print(f"Simple addition works: {z.shape}")

# Test nn.Module
import torch.nn as nn
linear = nn.Linear(10, 20).to(device)
output = linear(x)
print(f"Linear layer works: {output.shape}")

print("All tests passed!")

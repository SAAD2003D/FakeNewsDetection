import torch
from model import HCATModel

# Simulation: Assume we found these counts in dataset.py
num_spk = 3000
num_pty = 25
num_stt = 50
num_sbj = 100

print("Initializing Model...")
model = HCATModel(num_spk, num_pty, num_stt, num_sbj)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"✅ Model successfully moved to {device}")
print(model)
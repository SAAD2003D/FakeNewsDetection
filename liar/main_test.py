import torch

print("PyTorch Version:", torch.__version__)

# Check if CUDA (NVIDIA GPU support) is available
if torch.cuda.is_available():
    print("✅ Success! GPU is available.")
    print("  GPU Name:", torch.cuda.get_device_name(0))
    print("  Memory:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
else:
    print("❌ Error: GPU is NOT detected. Training will be very slow on CPU.")
    print("  Did you install the specific CUDA version of PyTorch?")
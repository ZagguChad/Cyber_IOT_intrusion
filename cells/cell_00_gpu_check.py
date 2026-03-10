# ======================================
# GPU CHECK & SYSTEM INFO
# ======================================
import torch
import subprocess, platform

print("=" * 50)
print("SYSTEM INFO")
print("=" * 50)
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No CUDA GPU detected. Models will run on CPU.")
print("=" * 50)

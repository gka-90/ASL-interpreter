import os
import torch

print("=== Environment Variables ===")
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"))
print("SLURM_JOB_ID:", os.environ.get("SLURM_JOB_ID", "Not Set"))

print("\n=== PyTorch CUDA Check ===")
cuda_available = torch.cuda.is_available()
print(f"Is CUDA available: {cuda_available}")

if cuda_available:
    device_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {device_count}")
    
    current_device = torch.cuda.current_device()
    print(f"Current active device index: {current_device}")
    
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Assigned GPU Name: {device_name}")
else:
    print("WARNING: PyTorch cannot see a GPU! Check your module/environment setup.")
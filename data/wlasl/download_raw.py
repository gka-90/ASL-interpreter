from huggingface_hub import snapshot_download
import os

# Define where you want to download the files (current directory)
download_dir = os.getcwd()

print(f"Downloading WLASL dataset files to: {download_dir}")

# Download the repository files from Hugging Face
snapshot_download(
    repo_id="Voxel51/WLASL",
    repo_type="dataset",
    local_dir=download_dir,
    local_dir_use_symlinks=False
)

print("Download complete!")

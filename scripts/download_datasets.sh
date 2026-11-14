#!/bin/bash
# Download WLASL and ASL Citizen to the HPC.
#
# Run on a login or data-transfer node (compute nodes often have no internet),
# inside tmux so it survives a dropped SSH connection:
#   tmux new -s download
#   bash scripts/download_datasets.sh            # both datasets
#   bash scripts/download_datasets.sh wlasl      # just one
#   bash scripts/download_datasets.sh citizen
#
# Space needed: ~6 GB for WLASL, ~90 GB peak for ASL Citizen
# (42.8 GB zip + unzipped copy; the zip is deleted afterwards).
#
# License note: both datasets are research-only. ASL Citizen forbids
# redistribution, so never commit these videos or extracted keypoints to git.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-${SCRATCH:-$HOME/scratch}/asl_data}"
WHICH="${1:-all}"
mkdir -p "$DATA_ROOT"
echo "Downloading into $DATA_ROOT"
df -h "$DATA_ROOT" | tail -1

download_wlasl() {
    # Hugging Face copy (FiftyOne format): ~5.5 GB, ~12k videos, labels in samples.json -> gloss.label
    python3 -m pip install --user --quiet --upgrade huggingface_hub
    export PATH="$HOME/.local/bin:$PATH"   # where --user installs the hf command
    local dest="$DATA_ROOT/wlasl"
    if command -v hf >/dev/null; then
        hf download Voxel51/WLASL --repo-type dataset --local-dir "$dest"
    else
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Voxel51/WLASL', repo_type='dataset', local_dir='$dest')"
    fi
    echo "WLASL done: $(find "$dest" -name '*.mp4' | wc -l) videos"
}

download_citizen() {
    # Microsoft ASL Citizen: 42.8 GB zip, 83,399 .mp4 videos, 2,731 signs, signer-independent splits
    local url="https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
    local zip="$DATA_ROOT/ASL_Citizen.zip"
    wget -c -O "$zip" "$url"          # -c resumes a partial download
    unzip -q -n "$zip" -d "$DATA_ROOT/asl_citizen"
    rm "$zip"
    echo "ASL Citizen done: $(find "$DATA_ROOT/asl_citizen" -name '*.mp4' | wc -l) videos"
}

case "$WHICH" in
    wlasl)   download_wlasl ;;
    citizen) download_citizen ;;
    all)     download_wlasl; download_citizen ;;
    *)       echo "Usage: $0 [all|wlasl|citizen]"; exit 1 ;;
esac

# Print the layout so the keypoint-extraction script can be written to match
echo; echo "=== Layout ==="
find "$DATA_ROOT" -maxdepth 3 -not -name '*.mp4' | head -40
find "$DATA_ROOT" -maxdepth 3 -name '*.csv' -exec sh -c 'echo "--- $1"; head -3 "$1"' _ {} \;

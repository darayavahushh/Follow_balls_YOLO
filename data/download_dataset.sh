#!/bin/bash
set -e

VENV_DIR=".venv"
DATA_DIR="data"
ZIP_FILE="$DATA_DIR/V1.zip"
EXTRACT_DIR="$DATA_DIR"


if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

echo "Installing huggingface_hub..."
python -m pip install -q --upgrade pip
python -m pip install -q huggingface_hub

echo "Downloading dataset..."
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Adit-jain/Soccana_player_ball_detection_v1",
    local_dir="data",
    repo_type="dataset"
)
EOF

echo "Extracting V1.zip..."
mkdir -p "$EXTRACT_DIR"
unzip -oq "$ZIP_FILE" -d "$EXTRACT_DIR"

echo "Done. Dataset available in $EXTRACT_DIR/V1"
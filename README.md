# Ball Detection & Tracking

A comprehensive computer vision system for detecting and tracking a ball in 2D, 3D, and Bird's Eye View (BEV). This project is developed as part of a Computer Vision challenge demonstrating object detection, tracking, and 3D reconstruction skills.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Create Virtual Environment](#2-create-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Download Dataset](#4-download-dataset)
- [Usage](#-usage)
  - [1. Dataset Preparation](#1-dataset-preparation)
  - [2. Configuration](#2-configuration)
  - [3. Training](#3-training)
  - [4. Inference](#4-inference)
  - [5. Output files](#5-output-files)
- [Troubleshooting](#-troubleshooting)
- [References](#-references)

---

## 🎯 Overview

This project processes RGB video footage to:

1. **2D Detection** - Detect the ball and persons in each frame using YOLO26
2. **3D Detection** - Estimate the ball's 3D position (x, y, z) relative to the camera *(coming soon)*
3. **Trajectory Tracking** - Track and visualize the ball's trajectory over time *(coming soon)*
4. **Bird's Eye View (BEV)** - Generate a top-down view map of ball positions *(coming soon)*

The system is designed for the Intel RealSense D435i camera and uses a standard FIFA size 5 football (22cm diameter) as a reference for depth estimation.

---

## ✨ Features

- **State-of-the-art Detection**: Uses YOLO26 for fast and accurate object detection
- **Configurable Pipeline**: All parameters centralized in `config.yaml`
- **Modular Design**: Each component (2D, 3D, BEV) works independently
- **Real-time Capable**: YOLO26s achieves ~400 FPS on GPU
- **Export Formats**: Outputs annotated videos and JSON detection data

---

## 📦 Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1060 (6GB) | NVIDIA RTX 3070+ |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 20 GB free |

> **Note**: CPU-only training is possible but significantly slower.

### Software Requirements

- Python 3.8 or higher
- CUDA 11.7+ (for GPU acceleration)
- Git

---

## 🚀 Installation

### 1. Clone Repository

```
git clone https://github.com/darayavahushh/Follow_balls_YOLO.git
cd Follow_balls_YOLO
```

### 2. Create Virtual Environment
We use a dedicated virtual environment named ball_cv to isolate project dependencies.
Using venv:
```
python -m venv ball_cv
source ball_cv/bin/activate  # Linux/Mac
ball_cv\Scripts\activate     # Windows
```
Using conda:
```
conda create -n ball_cv python=3.10
conda activate ball_cv
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Download Dataset
Download the soccer dataset from HuggingFace:
Option 1: Run the scrips located in ```data/```
- For Linux:
  ```
  chmod +x .data/download_dataset.sh
  ./data/download_dataset.sh
  ```
- For Windows:
  ```
  Double click or run in cmd ./data/download_dataset.bat
  ```

Option 2: Using git
```
# Make sure git-xet is installed (https://hf.co/docs/hub/git-xet)
winget install git-xet

git clone https://huggingface.co/datasets/Adit-jain/Soccana_player_ball_detection_v1

# Unzip the V1.zip and store it in ./data/V1
```

---

## 🎮 Usage

### 1. Dataset Preparation

The training script automatically preprocesses the dataset:

- Remaps classes (player+referee → person, ball → ball)
- Creates train/val split
- Generates data.yaml for YOLO

### 2. Configuration

All project settings are in configs/config.yaml with complete documentation of all options.

### 3. Training (creates new run)

```
# Basic training (uses config.yaml defaults)
python src/train_2d.py --config configs/config.yaml

# Override specific parameters
python src/train_2d.py --config configs/config.yaml --epochs 50 --batch-size 8

# Skip preprocessing (if already done)
python src/train_2d.py --config configs/config.yaml --skip-preprocessing
```

### 4. Inference (uses existing run)

```
# Run detection on video
python src/inference_2d.py --config configs/config.yaml --run-id run_001

# Ball-only detection (ignore persons)
python src/inference_2d.py --config configs/config.yaml --run-id run_001 --ball-only

# Disable preview window
python src/inference_2d.py --config configs/config.yaml --run-id run_001 --no-preview

# Use specific model weights
python src/inference_2d.py --config configs/config.yaml --run-id run_001 --model path/to/custom/weights.pt
```

#### List Available Runs

```
python -c "from tools.run_manager import RunManager; print(RunManager.list_runs())"
```

### 5. Output Files

**Annotated Video**: ```outputs/results/rgb_2d_detected.avi```

- Original video with bounding boxes drawn
- Green boxes for ball, blue boxes for persons
- Frame counter and ball detection status overlay

**Detections JSON**: ```outputs/results/rgb_detections.json```

---

## 🔧 Troubleshooting

### CUDA Out of Memory

- Reduce batch size in config.yaml or via CLI
    ```
    python src/train_2d.py --config configs/config.yaml --batch-size 8
    ```

### Model Not Found

- Ensure you're using correct model name
- Valid options: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt

### Video Cannot Be Opened

- Check video path in config.yaml
- Ensure video codec is supported (install ffmpeg if needed)
    ```
    pip install opencv-python-headless
    ```

Low Detection Accuracy

- In config.yaml, try:
    ```
    model:
        confidence_threshold: 0.2  # Lower threshold
    training:
        epochs: 150  # More epochs
    ```

---

## 📚 References

- [Intel RealSense D435i Camera](https://www.intelrealsense.com/depth-camera-d435i/)
- [Intel RealSense Beginner's Guide to Depth](https://www.intelrealsense.com/beginners-guide-to-depth/)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/models)
- [Soccer Detection Dataset on HuggingFace](https://huggingface.co/datasets/Adit-jain/Soccana_player_ball_detection_v1)

---

## 📄 License
This project is developed for educational purposes as part of a Computer Vision challenge.

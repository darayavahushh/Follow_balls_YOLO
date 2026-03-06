# Ball Detection & Tracking

A comprehensive computer vision system for detecting and tracking a football in 2D, 3D, and Bird's Eye View (BEV). This project is developed as part of a Computer Vision challenge demonstrating object detection, tracking, and 3D reconstruction skills.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
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

1. **2D Detection** - Detect the football and persons in each frame using YOLO26
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

## 📁 Project Structure

```text
football_cv/
├── configs/
│ └── config.yaml                # Centralized configuration
│
├── src/
│ ├── init.py                    # Package initialization
│ ├── train_2d.py                # 2D detection training
│ ├── inference_2d.py            # 2D detection inference
│ ├── detection_3d.py            # 3D position estimation (coming soon)
│ ├── trajectory.py              # Trajectory tracking (coming soon)
│ └── bev.py                     # Bird's eye view (coming soon)
│
├── data/
│ └── V1/                        # Dataset from HuggingFace
│ │ ├── images/
│ │ │ ├── train/
│ │ │ └── test/
│ │ ├── labels/
│ │ │ ├── train/
│ │ │ └── test/
│ └── data.yaml
│
├── outputs/
│ ├── datasets/                  # Processed datasets
│ │ └── processed/
│ ├── models/                    # Trained model weights
│ │ ├── train_run/
│ │ ├── weights/
│ │ ├── best.pt
│ │ └── last.pt
│ └── results/                   # Inference outputs
│ │ ├── rgb_2d_detected.avi
│ │ └── rgb_detections.json
│
├── rgb.avi                      # Input video
├── requirements.txt             # Python dependencies
└── README.md
```

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
Option 1: Using huggingface_hub
```
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Adit-jain/Soccana_player_ball_detection_v1', local_dir='data/V1', repo_type='dataset')"
```
Option 2: Using git
```
# Make sure git-xet is installed (https://hf.co/docs/hub/git-xet)
winget install git-xet

git clone https://huggingface.co/datasets/Adit-jain/Soccana_player_ball_detection_v1
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

### 3. Training

```
# Basic training (uses config.yaml defaults)
python src/train_2d.py --config configs/config.yaml

# Override specific parameters
python src/train_2d.py --config configs/config.yaml --epochs 50 --batch-size 8

# Skip preprocessing (if already done)
python src/train_2d.py --config configs/config.yaml --skip-preprocessing
```

### 4. Inference

```
# Run detection on video
python src/inference_2d.py --config configs/config.yaml

# Ball-only detection (ignore persons)
python src/inference_2d.py --config configs/config.yaml --ball-only

# Disable preview window
python src/inference_2d.py --config configs/config.yaml --no-preview

# Use specific model weights
python src/inference_2d.py --config configs/config.yaml --model path/to/custom/weights.pt
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

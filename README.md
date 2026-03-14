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
  - [4. 2D Inference](#4-2d-inference)
  - [5. 3D Detection & Trajectory](#5-3d-detection--trajectory)
  - [6. 2D Top-View Map (BEV)](#6-2d-top-view-map-bev)
  - [7. Output Files](#7-output-files)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [References](#-references)

---

## 🎯 Overview

This project processes RGB video footage to:

1. **2D Detection** — Detect the ball and persons in each frame using YOLO26
2. **3D Detection** — Estimate the ball's 3D position (X, Y, Z) and distance relative to the camera using the pinhole camera model and known ball diameter
3. **Trajectory Tracking** — Track and visualize the ball's trajectory with a fading trail overlay
4. **Bird's Eye View (BEV)** — Generate a camera-independent top-down 2D map via ego-motion estimation, displayed as a split-view video alongside the original footage

The system is designed for the Intel RealSense D435i camera and uses a standard FIFA size 5 football (22cm diameter) as a reference for depth estimation.

---

## ✨ Features

- **State-of-the-art Detection**: Uses YOLO26 for fast and accurate object detection
- **3D Position Estimation**: Pinhole camera model + known football diameter → depth, 3D coordinates, and Euclidean distance
- **Trajectory Visualization**: Fading polyline trail with optional smoothing drawn on the video
- **Camera-Independent BEV Map**: Sparse optical flow + Essential-matrix ego-motion estimation transforms the ball into a fixed world frame unaffected by camera shake
- **Split-View Output**: Side-by-side video (annotated footage | 2D top-view map) for easy interpretation
- **Configurable Pipeline**: All parameters centralized in `config.yaml`
- **Modular Design**: Each component (2D, 3D, BEV) works independently
- **Export Formats**: Annotated videos, JSON detection data, and Excel spreadsheets with 3D/world positions

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
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
Using conda:
```
conda create -n venv python=3.10
conda activate venv
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

### 4. 2D Inference

Runs the trained YOLO model on a video. Outputs an annotated video and a JSON file with per-frame detections.

```
# Run 2D detection on video
python src/inference_2d.py --config configs/config.yaml --run-id run_001

# Ball-only detection (ignore persons)
python src/inference_2d.py --config configs/config.yaml --run-id run_001 --ball-only

# Disable preview window
python src/inference_2d.py --config configs/config.yaml --run-id run_001 --no-preview
```

### 5. 3D Detection & Trajectory

Estimates the ball's 3D position (X, Y, Z) and distance relative to the camera for every frame, and draws a fading trajectory trail on the video. Also exports an Excel spreadsheet with per-frame 3D coordinates.

The depth is computed using the **pinhole camera model**: the ball's apparent pixel diameter compared to its real-world diameter (22 cm) gives the depth along the optical axis. The 3D position follows from back-projecting the bounding-box centre through the camera intrinsics.

```
# Run 3D detection + trajectory
python src/inference_3d.py --config configs/config.yaml --run-id run_001

# Without preview
python src/inference_3d.py --config configs/config.yaml --run-id run_001 --no-preview
```

### 6. 2D Top-View Map (BEV)

Produces a **split-view video**: the left panel shows the annotated footage (bounding boxes + 3D overlay + trajectory), and the right panel shows a 2D top-view map with the ball and camera positions in a **fixed world frame** that is not affected by camera movement.

Camera ego-motion is estimated per-frame using:
1. Sparse optical flow (Lucas-Kanade) to track background features
2. Essential matrix decomposition (RANSAC) to recover rotation and translation
3. Scale resolution using the ball's estimated depth
4. Cumulative pose accumulation to build a world coordinate frame

```
# Run BEV split-view
python src/inference_bev.py --config configs/config.yaml --run-id run_001

# Without preview
python src/inference_bev.py --config configs/config.yaml --run-id run_001 --no-preview
```

#### List Available Runs

```
python -c "from tools.run_manager import RunManager; print(RunManager.list_runs())"
```

### 7. Output Files

All outputs are saved under `outputs/runs/<run_id>/results/`.

| Pipeline | File | Description |
|----------|------|-------------|
| 2D | `rgb_2d_detected.avi` | Video with bounding boxes, frame counter, ball status |
| 2D | `rgb_detections.json` | Per-frame detection data (bboxes, classes, confidence) |
| 3D | `rgb_3d_trajectory.avi` | Video with 3D overlay + trajectory trail |
| 3D | `rgb_3d_positions.xlsx` | Excel: frame, time, X, Y, Z, distance vs time |
| 3D | `rgb_3d_detections.json` | Full detection + 3D data per frame |
| BEV | `rgb_bev_split.avi` | Split-view video: [annotated \| top-view map] |
| BEV | `rgb_bev_positions.xlsx` | Excel: camera + ball positions in both camera and world frame |
| BEV | `rgb_bev_detections.json` | Full detection + 3D + world-frame data per frame |

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

## � Project Structure

```
Follow-balls-YOLO/
├── configs/
│   └── config.yaml              # All configurable parameters
├── src/
│   ├── train_2d.py              # Training pipeline (dataset prep + YOLO fine-tuning)
│   ├── inference_2d.py          # 2D detection inference
│   ├── inference_3d.py          # 3D detection + trajectory tracking
│   └── inference_bev.py         # BEV split-view (top-down map)
├── tools/
│   ├── config_loader.py         # YAML config loading & path resolution
│   ├── visualization.py         # Bounding box & overlay drawing
│   ├── video_io.py              # Video reader/writer with context managers
│   ├── detection_utils.py       # Detection data extraction & formatting
│   ├── logging_utils.py         # Dual console/file logging
│   ├── run_manager.py           # Experiment run management (run_001, run_002, …)
│   ├── depth_estimation.py      # Pinhole-model 3D position estimation
│   ├── trajectory.py            # Ball trajectory tracking & trail drawing
│   └── bev_map.py               # Camera ego-motion + BEV map rendering
├── data/                        # Dataset (downloaded separately)
├── outputs/                     # All outputs (runs, models, results)
└── requirements.txt
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

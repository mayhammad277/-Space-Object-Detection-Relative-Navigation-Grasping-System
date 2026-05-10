# 🌌 Space Object Detection, Relative Navigation & Grasping System




[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular computer vision system for **autonomous space object detection, relative position estimation, and visual servoing for robotic grasping** in orbital environments.

Integrates **star tracker attitude determination** with **real-time target tracking** and **closed-loop approach control**.

---

## 🎯 System Overview
```text

┌─────────────────────────────────────────────────────────────┐
│ SPACE GRASPING SYSTEM │
├───────────────┬──────────────────┬──────────────────────────┤
│ Star Tracker │ Object Detection │ Visual Servoing │
│ (Attitude) │ (YOLO/Color/ORB) │ (Approach Control) │
├───────────────┴──────────────────┴──────────────────────────┤
│ Position Estimator (Distance & Angle) │
├─────────────────────────────────────────────────────────────┤
│ Telemetry Handler (UART/Serial) │
└─────────────────────────────────────────────────────────────┘


```




```text

space-grasping-system/
├── src/                        # Core Logic & Algorithms
│   ├── star_tracker.py         # Star field matching & attitude determination
│   ├── object_detector.py      # Multi-method (DL/Geometric) detection
│   ├── position_estimator.py   # 6-DOF distance and angle calculation
│   ├── visual_servoing.py      # IBVS/PBVS approach control loops
│   ├── telemetry.py            # Serial communication & data logging
│   └── utils.py                # Math & coordinate transformations
│
├── config/                     # System Configuration
│   ├── params.yaml             # PID gains and detector thresholds
│   └── calibration.yaml        # Intrinsic/extrinsic camera parameters
│
├── data/                       # Resources & Assets
│   ├── templates/              # Star catalog & constellation maps
│   ├── reference/              # 3D models & reference object images
│   └── calibration/            # Checkerboard images for CV setup
│
├── models/                     # Weights for ML-based detection
│
├── notebooks/                  # Development & R&D
│   ├── 01_camera_calibration.ipynb
│   ├── 02_object_detection_test.ipynb
│   └── 03_system_integration_test.ipynb
│
├── tests/                      # Validation Suite
│   ├── test_detector.py
│   ├── test_position_estimator.py
│   └── test_visual_servoing.py
│
├── main.py                     # Application entry point
├── requirements.txt            # Environment dependencies
├── setup.py                    # Package installation script
├── .gitignore                  # Version control exclusions
├── LICENSE                     # Project licensing
└── README.md                   # Project documentation
```



### Key Features

- **Multi-method object detection**: YOLOv8, HSV color thresholding, ORB feature matching
- **Monocular distance estimation**: Using known object width and focal length
- **Stereo vision support**: Configurable baseline for metric depth
- **Star tracker integration**: FFT-based attitude determination from star fields
- **Visual servoing controller**: Proportional control for autonomous approach
- **Real-time telemetry**: Structured serial output for downstream systems
- **Smoothing filters**: Exponential moving average for robust position estimates
- **Modular architecture**: Each subsystem independently testable

---

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/mayhammad227/space-grasping-system.git
cd space-grasping-system

# Install dependencies
pip install -r requirements.txt
```






## Basic Usage 

```bash
# Run with webcam (default)
python main.py

# Run with video file
python main.py --video fl3.mp4

# Run with custom configuration
python main.py --config config/params.yaml --video 0

# Run with star tracker integration
python main.py --video fl3.mp4 --star-tracker


```



📊 System Modes
- Mode	Description	Behavior
- SCANNING	Searching for target	Rotates slowly to cover FOV
- TRACKING	Target acquired	Approaches while keeping target centered
- GRASPING	Final approach	Executes precision grasp sequence
- ORIENTING	Star tracker active	Computes absolute attitude
- HOLDING	Position hold	Maintains current state
- ERROR	Target lost	Safety stop, re-enters scanning




🔧 Hardware Setup

```text


Monocular Mode

Camera (USB/CSI) → Compute → Control Commands → Robotic Arm





Stereo Mode

Left Camera ─┐
             ├→ Stereo Depth → Compute → Control Commands
Right Camera ┘


With Star Tracker



Star Camera ─→ Attitude ─┐
                          ├→ Sensor Fusion → Navigation Solution
Object Camera ─→ Position ┘

## Pre-Processing Pipeline

```

```text


Frame Input
    │
    ▼
┌─────────────────┐
│ Object Detection │ ─── YOLO / Color / ORB
└────────┬────────┘
         │ bbox, centroid, width_px
         ▼
┌─────────────────────┐
│ Position Estimation │ ─── Distance + Angle
└────────┬────────────┘
         │ distance_cm, angle_deg
         ▼
┌──────────────────────┐
│ Visual Servoing      │ ─── PID Control
│ (Approach Controller)│
└────────┬─────────────┘
         │ linear_vel, angular_vel
         ▼
┌─────────────────┐
│ Motor Commands   │ ─── Robotic Arm / Platform
└─────────────────┘


```


📡 Telemetry Protocol


```text


dang,<angle_deg>,dist,<distance_cm>,mode,<system_mode>,scam,<star_constellation>,x,<checksum>


```



🎮 Control Logic
## Object Detection Decision Tree

```text
Is target detected?
├── YES → Lock target → Estimate position → Compute approach velocity
└── NO  → Count lost frames
          ├── Lost < threshold → Use last known position (coast)
          └── Lost > threshold → Enter SCANNING mode
```

## Grasping Conditions
```text
Ready to grasp when:
  1. target_aligned == True      (angle error < 2°)
  2. distance <= grasp_dist + 3cm
  3. angular_error < alignment_threshold
```






<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e2ea9922-53f0-441a-bf37-296fc52fbb64" />


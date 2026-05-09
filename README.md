# 🌌 Space Object Detection, Relative Navigation & Grasping System




[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular computer vision system for **autonomous space object detection, relative position estimation, and visual servoing for robotic grasping** in orbital environments.

Integrates **star tracker attitude determination** with **real-time target tracking** and **closed-loop approach control**.

---

## 🎯 System Overview

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









space-grasping-system/
├── src/
│   ├── __init__.py              # Package init
│   ├── star_tracker.py          # Star tracking & attitude
│   ├── object_detector.py       # Multi-method detection
│   ├── position_estimator.py    # Distance & angle
│   ├── visual_servoing.py       # Approach controller
│   ├── telemetry.py             # Serial telemetry
│   └── utils.py                 # Shared utilities
├── config/
│   ├── params.yaml              # Default configuration
│   └── calibration.yaml         # Camera calibration
├── data/
│   ├── templates/               # Star field templates
│   ├── reference/               # Reference object images
│   └── calibration/             # Calibration images
├── models/                      # Pre-trained weights
├── notebooks/
│   ├── 01_camera_calibration.ipynb
│   ├── 02_object_detection_test.ipynb
│   └── 03_system_integration_test.ipynb
├── tests/
│   ├── test_detector.py
│   ├── test_position_estimator.py
│   └── test_visual_servoing.py
├── main.py                      # Entry point
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
└── README.md


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

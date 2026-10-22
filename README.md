# 🚗 Autonomous Self-Driving Car: End-to-End Perception, Control & Explainable AI

[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)](https://www.python.org/)
[![Webots](https://img.shields.io/badge/Simulator-Webots%20R2023a-red.svg)](https://cyberbotics.com/)
[![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%2F%20Keras-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-yellow.svg)](https://scikit-learn.org/)

An end-to-end autonomous driving platform combining **Robotics Simulation (Webots)**, **Classical Computer Vision**, **Machine Learning & Deep Learning (HOG+SVM, CNN Behavioral Cloning)**, **2D LiDAR Collision Avoidance**, and **Explainable AI (LIME)**.

---

## 📑 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [High-Level Design (HLD) & System Architecture](#-high-level-design-hld--system-architecture)
3. [Key Modules & Technical Deep-Dive](#-key-modules--technical-deep-dive)
   - [1. Simulation Environment & Vehicle Model](#1-simulation-environment--vehicle-model)
   - [2. Teleoperation & Dataset Generation](#2-teleoperation--dataset-generation)
   - [3. Vision-Based Lane Keeping (Color Thresholding + Proportional Control)](#3-vision-based-lane-keeping)
   - [4. Road Sign Classification (Classical ML vs. Deep Learning)](#4-road-sign-classification)
   - [5. End-to-End Behavioral Cloning (CNN Steering Prediction)](#5-end-to-end-behavioral-cloning)
   - [6. LiDAR-Based Collision Avoidance & Navigation](#6-lidar-based-collision-avoidance)
   - [7. Explainable AI (LIME Superpixel Saliency Maps)](#7-explainable-ai-lime)
4. [Technology Stack & Tools](#-technology-stack--tools)
5. [Repository Structure](#-repository-structure)
6. [Installation & Setup](#-installation--setup)
7. [Step-by-Step Execution Guide](#-step-by-step-execution-guide)
8. [Performance Benchmarks & Results](#-performance-benchmarks--results)
9. [How to Pitch & Describe This Project](#-how-to-pitch--describe-this-project)

---

## 🌟 Executive Summary

This project implements a full-stack autonomous vehicle perception and navigation system tested inside a 3D urban robotics environment (**Webots**). It solves core autonomous driving challenges through multiple complementary paradigms:

- **Perception:** Dual-stream perception combining camera imagery for lane/sign detection and 2D LiDAR range imaging for obstacle clearance.
- **Control:** Proportional lane centering from image moments, alongside deep convolutional regression/classification for autonomous end-to-end steering.
- **Robustness:** Addressing extreme class imbalance with SMOTE, benchmarking HOG + Support Vector Machines against Deep Convolutional Neural Networks.
- **Safety & Collision Avoidance:** Bird's-Eye View (BEV) Cartesian projection from polar LiDAR point clouds with real-time waypoint steering.
- **Interpretability:** Explainable AI (XAI) using LIME to inspect superpixel activations and verify what the autonomous driving CNN actually sees.

---

## 🏗️ High-Level Design (HLD) & System Architecture

### Architectural Overview

```mermaid
flowchart TD
    subgraph Simulation ["Webots 3D Simulation Environment (BMW X5)"]
        Sensors["Sensors: Front RGB Camera + SICK LMS 291 2D LiDAR"]
        Actuators["Vehicle Actuators: Steering Angle & Cruising Speed (Driver API)"]
        VirtualHUD["In-Sim Virtual HUD Display (Display API)"]
    end

    subgraph DataCollection ["Human Teleoperation & Data Logging"]
        InputDevices["Xbox Controller (inputs) / Keyboard (pynput)"]
        Recorder["1_create_dataset.py: Synchronized Frame & Steering Logger"]
        DatasetStore[("Dataset: PNG Frames + JSON Metadata")]
    end

    subgraph PerceptionSign ["Road Sign Detection & Classification"]
        RawSignData[("Kaggle Road Sign Dataset: PASCAL VOC XML")]
        SignPreproc["preprocessing.py: XML Parser + Square Cropper + Resize"]
        HOG_SVM["Classical ML: Grayscale -> HOG Feature Extractor -> SVM (RBF)"]
        CNN_Sign["Deep Learning: SMOTE Resampling -> Keras CNN Classifier"]
    end

    subgraph LaneKeeping ["Classical Computer Vision Lane Following"]
        CamFrame["Live RGB Camera Frame"]
        ColorMask["utils.colour_threshold_cv2: HSV Yellow Masking"]
        Centroid["utils.calculate_normalized_average_col: Center of Mass"]
        P_Controller["Proportional Steering Controller (2_camera_pid.py)"]
    end

    subgraph BehavioralCloning ["End-to-End Neural Behavioral Cloning"]
        TrainPipeline["3_train.py (Classification) / 3_train_reg.py (Regression)"]
        ModelArtifact[("Trained CNN (.h5 Model)")]
        InferenceEngine["4_run_model.py: Real-time Live Steering Predictor"]
    end

    subgraph CollisionAvoidance ["LiDAR Active Collision Avoidance"]
        RawLiDAR["Raw LiDAR Range Array (180 deg FOV, 180 rays)"]
        PolarToCart["Polar to Cartesian Transform (BEV Projection 64x128)"]
        HorizonScan["ROW_HIT Scanline Obstacle Detection"]
        WaypointGen["Corridor Midpoint Calculation & Steering Correction"]
    end

    subgraph Explainability ["Model Interpretability (XAI)"]
        LIME_Engine["5_explain.ipynb: LIME Image Explainer"]
        SaliencyMap["Superpixel Saliency Masks (Positive/Negative Evidence)"]
    end

    %% Connections
    Sensors -->|Live RGB Feed| CamFrame
    Sensors -->|Live RGB Feed| Recorder
    Sensors -->|Live LiDAR Ranges| RawLiDAR
    InputDevices -->|Manual Teleop| Recorder
    InputDevices -->|Manual Teleop| Actuators
    Recorder --> DatasetStore

    CamFrame --> ColorMask --> Centroid --> P_Controller --> Actuators
    Centroid --> VirtualHUD

    RawSignData --> SignPreproc --> HOG_SVM
    SignPreproc --> CNN_Sign

    DatasetStore --> TrainPipeline --> ModelArtifact
    Sensors -->|Live Camera Feed| InferenceEngine
    ModelArtifact --> InferenceEngine --> Actuators

    RawLiDAR --> PolarToCart --> HorizonScan --> WaypointGen --> Actuators
    PolarToCart --> VirtualHUD

    ModelArtifact --> LIME_Engine
    DatasetStore --> LIME_Engine --> SaliencyMap
```

### Subsystem Interaction Matrix

| Subsystem | Input Modality | Processing Method | Output | Primary Files |
|---|---|---|---|---|
| **Human Teleoperation** | Gamepad / Keyboard | Threaded non-blocking polling, dead-zone filter | Normalized Throttle & Steering angle | `driving_inputs.py`, `1_create_dataset.py` |
| **CV Lane Keeping** | RGB Camera ($64 \times 128$) | HSV yellow color space thresholding, centroid moment | Normalized deviation $[-0.5, 0.5] \to$ Steering | `2_computer_vision/2_camera_pid.py`, `utils.py` |
| **Road Sign ML** | $100 \times 100$ Cropped Signs | HOG descriptors (9 orientations, $14\times14$ cells) + SVM | 4-class classification (`speedlimit`, `stop`, etc.) | `preprocessing.py`, `2_hog_svm.py` |
| **Road Sign CNN** | $100 \times 100$ RGB Signs | SMOTE balance + Multi-layer Conv2D/Dropout | Multi-class Softmax Probability | `3_cnn.ipynb`, `preprocessing.py` |
| **Behavioral Cloning** | Camera RGB Stream | End-to-End CNN (Conv2D $\to$ MaxPool $\to$ Dense $\to$ Softmax/Linear) | Discrete Direction (L/S/R) or Continuous Angle | `3_train.py`, `3_train_reg.py`, `4_run_model.py` |
| **LiDAR Obstacle Avoidance** | SICK LMS 291 1D Range Array | Polar-to-Cartesian BEV projection, corridor waypoint finding | Steering offset & Collision warning | `4_collision_avoidance/2_lidar_automated.py` |
| **Explainable AI (XAI)** | Road Driving Images | LIME Perturbation + Ridge regression on superpixels | Visualized salient road features | `5_explain.ipynb` |

---

## 🔍 Key Modules & Technical Deep-Dive

### 1. Simulation Environment & Vehicle Model
- **Simulator:** Cyberbotics Webots R2023a simulation engine.
- **Worlds:**
  - `worlds/city.wbt`: Comprehensive city layout with cross-sections, traffic lights, road markings, buildings, and vegetation.
  - `worlds/city_with_borders.wbt`: Bounded urban track for safety envelope testing.
- **Vehicle Rig:** Simulated BMW X5 fitted with:
  - Color camera mounted on front bumper/windshield.
  - SICK LMS 291 2D LiDAR scanner mounted at the front bumper level.
  - Interactive Virtual Display (`display_th`) acting as an onboard Heads-Up Display (HUD).

### 2. Teleoperation & Dataset Generation
- **Controller Abstraction (`driving_inputs.py`):**
  - Threaded asynchronous monitor for Xbox gamepads (`inputs` library) with analog dead-zone compensation.
  - Non-blocking keyboard fallback (`pynput`) supporting Arrow keys and WASD with smooth ramp rates.
- **Data Collection Loop (`1_create_dataset.py`):**
  - Captures synchronized pairs of driver actions and visual states.
  - Automatically suppresses idle stationary frames (`throttle == 0`) to eliminate zero-velocity dataset poisoning.
  - Saves frames as PNG images with paired JSON metadata (`angle`, `throttle`, `image_path`).

### 3. Vision-Based Lane Keeping
- **HSV Segmentation (`2_computer_vision/`):**
  - Transforms raw RGB camera frames into HSV color space, which provides illumination invariance against shadows and glare.
  - Applies yellow road-marker threshold mask (`yellow_pixel = [25, 127, 127]`, error bounds).
- **Center-of-Mass Tracking:**
  - Computes $\text{Col}_{\text{avg}} = \frac{1}{N} \sum_{i=1}^N \text{Col}_i$ across all positive mask pixels.
  - Normalizes the error metric between $[-0.5, 0.5]$ relative to frame center.
- **Proportional Control:**
  - Directly commands vehicle steering angle proportional to lane deviation, continuously centering the vehicle in the lane.
  - Overlays binary mask and tracked centroid onto the in-sim HUD in real time.

### 4. Road Sign Classification: Classical ML vs. Deep Learning
- **Dataset Pipeline (`preprocessing.py`):**
  - Parses PASCAL VOC XML annotations from Kaggle Road Sign Detection dataset.
  - Crops bounding boxes with automated aspect-ratio squaring and bounds-clipping algorithm.
  - Discards microscopic bounding boxes ($< 20\text{px}$) to avoid noise artifacts.
- **Classical HOG + SVM Pipeline (`2_hog_svm.py`):**
  - Scikit-Learn pipeline chaining `RGB2GrayTransformer`, `HogTransformer` (9 orientations, $14 \times 14$ pixels/cell, $2 \times 2$ blocks, L2-Hys norm), and an RBF-kernel `SVC`.
  - Achieves **~98.5% – 99% test accuracy**.
- **Deep CNN with SMOTE (`3_cnn.ipynb`):**
  - Solves severe real-world class imbalance (`speedlimit` outnumbered other signs ~7:1) using **SMOTE** (Synthetic Minority Over-sampling Technique).
  - Trains a deep CNN with $7\times7$ and $3\times3$ convolutions, Dropout regularization, EarlyStopping, and TensorBoard logging.

### 5. End-to-End Behavioral Cloning
- **Imitation Learning Architecture (`3_train.py` & `3_train_reg.py`):**
  - **Input:** $64 \times 128 \times 3$ normalized RGB highway/city frame.
  - **Feature Extractor:** 5 convolutional layers (filters 64, 128, 128, 256, 256) with ReLU activations and MaxPooling.
  - **Fully Connected Head:** Flatten $\to$ Dense(128) $\to$ Dropout(0.2) $\to$ Dense(64) $\to$ Dropout(0.2).
  - **Output Heads:**
    - *Classification:* 3-way Softmax (`Left`, `Straight`, `Right`).
    - *Regression:* Linear activation predicting steering radians directly ($[-0.25, 0.25]$ rad).
- **Runtime Autopilot Engine (`4_run_model.py`):**
  - Feeds live camera buffers into the pre-trained neural network at native simulation frequency.
  - Seamless human override: manual controller inputs immediately take precedence if human intervention occurs.

### 6. LiDAR-Based Collision Avoidance
- **Polar-to-Cartesian Projection (`4_collision_avoidance/`):**
  - The SICK LMS 291 emits 180 range readings across a $180^\circ$ forward arc.
  - Converts polar measurements $(r_i, \theta_i)$ to Cartesian coordinates $(x_v, y_v)$ in the vehicle reference frame:
    $$x_v = r \cdot \cos(\theta), \quad y_v = r \cdot \sin(\theta)$$
  - Projects coordinates into a $64 \times 128$ Bird's-Eye View (BEV) occupancy grid.
- **Autonomous Evasion Algorithm (`2_lidar_automated.py`):**
  - Evaluates an active collision horizon row (`ROW_HIT = 32`).
  - Detects obstacle hit clusters, isolates the closest obstruction boundaries, and dynamically calculates the safest gap midpoint.
  - Computes corrective steering angle $\theta_{\text{steer}} = (\text{Waypoint}_{\text{col}} - 64) \times 0.1$, maneuvering the vehicle safely around static and dynamic obstacles.

### 7. Explainable AI (LIME)
- **Interpretability via LIME (`5_explain.ipynb`):**
  - Implements Local Interpretable Model-agnostic Explanations (`lime.lime_image.LimeImageExplainer`).
  - Segments road scenes into superpixels and generates 1,000 perturbed sample variations.
  - Computes ridge-regression surrogate models to produce saliency masks highlighting:
    - **Green Superpixels:** Image regions positively reinforcing steering predictions (e.g. road edges, lane borders).
    - **Red Superpixels:** Regions opposing or distracting the prediction.
  - Validates that the network makes decisions based on legitimate road geometry rather than simulation artifacts.

---

## 🛠️ Technology Stack & Tools

| Category | Technology / Library | Purpose |
|---|---|---|
| **Simulation** | Webots R2023a, `vehicle.Car`, `vehicle.Driver` | 3D Physics simulation, vehicle dynamics, sensor feeds |
| **Core Languages** | Python 3.8+ | Primary programming language |
| **Computer Vision** | OpenCV (`cv2`), Scikit-Image (`skimage`), PIL | Image filtering, HSV segmentation, HOG extraction, BEV plotting |
| **Machine Learning** | Scikit-Learn (`sklearn`), Imbalanced-Learn (`imblearn`) | SVM classifier, pipelines, SMOTE oversampling, evaluation metrics |
| **Deep Learning** | TensorFlow 2.x, Keras, TensorBoard | CNN architecture, training loops, EarlyStopping, TensorBoard profiling |
| **Explainable AI** | LIME (`lime.lime_image`) | Superpixel perturbation and visual attribution maps |
| **Hardware & Input** | `inputs`, `pynput`, `threading` | Asynchronous multi-threaded gamepad and keyboard teleoperation |
| **Profiling & Math** | NumPy, `cProfile`, `pstats` | High-performance vector math and CPU bottleneck profiling |

---

## 📁 Repository Structure

```
Self-Driving-car-main/
├── 0_install/                         # Environment and installation configs
│   ├── general.yml                    # Conda environment definition
│   ├── install_linux.sh               # Automated Linux setup script
│   ├── requirements.txt               # PIP package dependencies
│   └── README.md                      # Installation instructions
├── 1_create_dataset.py                # Teleoperation recorder (Camera + Gamepad/Keyboard -> JSON/PNG)
├── 1_hog.ipynb                        # HOG feature extraction exploratory notebook
├── 2_computer_vision/                 # Classical vision lane keeping module
│   ├── 1_image_threshold.py           # Standalone HSV color thresholding test
│   ├── 2_camera_pid.py                # Closed-loop camera lane follower with virtual HUD
│   ├── image_from_sim.png             # Sample simulation test frame
│   ├── utils.py                       # Vision utilities (HSV threshold, centroid calculation)
│   └── README.md                      # Computer vision guide
├── 2_dataset_insights.ipynb           # EDA, label distribution, and SMOTE oversampling analysis
├── 2_hog_svm.py                       # Classical ML Road Sign classifier (HOG + RBF SVM)
├── 3_cnn.ipynb                        # Deep CNN Road Sign classifier with SMOTE balance
├── 3_train.py                         # Behavioral Cloning CNN (Classification: Left/Straight/Right)
├── 3_train_reg.py                     # Behavioral Cloning CNN (Regression: Continuous Angle)
├── 4_collision_avoidance/             # LiDAR perception and collision avoidance
│   ├── 1_lidar_manual.py              # Manual driving with real-time BEV LiDAR visualizer
│   ├── 2_lidar_automated.py           # Fully automated LiDAR corridor steering & obstacle avoidance
│   └── driving_inputs.py              # Local teleop driver helper
├── 4_run_model.py                     # Live autonomous driving controller running trained CNN in Webots
├── 5_explain.ipynb                    # Explainable AI (LIME) superpixel visual attribution
├── driving_inputs.py                  # Multithreaded Xbox Gamepad and Keyboard input handler
├── preprocessing.py                   # Kaggle PASCAL VOC dataset parser & smart cropper
├── utils.py                           # Shared utilities (model loader, normalization, image conversion)
├── worlds/                            # Webots 3D simulation worlds and textures
│   ├── city.wbt                       # Full urban city simulation
│   ├── city_with_borders.wbt          # Track simulation with boundary fences
│   └── textures/                      # World ground surfaces and road textures
└── README.md                          # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
1. **Webots Simulator:** Download and install [Webots R2023a or newer](https://cyberbotics.com/).
2. **Python Environment:** Python 3.8 to 3.10 recommended (compatible with TensorFlow 2.x).

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Self-Driving-car.git
cd Self-Driving-car
```

### 2. Set Up Python Environment
Using Conda:
```bash
conda env create -f 0_install/general.yml
conda activate self-driving
```
Or using Pip:
```bash
pip install -r 0_install/requirements.txt
```

### 3. Configure Webots Python Controller Path
Ensure Webots recognizes your Python executable:
- In Webots: **Preferences $\to$ Python command** $\to$ set to your active python binary (e.g. `python` or `path/to/conda/envs/self-driving/python.exe`).

---

## 🚀 Step-by-Step Execution Guide

### Phase 1: Test Classical Computer Vision Lane Following
1. Open Webots and load `worlds/city.wbt`.
2. Set the controller of the BMW car to `2_camera_pid.py` (or execute via Webots IDE).
3. The car will autonomously track the yellow road lines using real-time HSV segmentation and centroid calculation.

### Phase 2: Collect Driving Data (Human-in-the-Loop)
1. Run `1_create_dataset.py` as the Webots controller.
2. Drive the car around the city using an **Xbox Controller** (Left Stick: Throttle, Right Stick: Steering) or **Keyboard** (Arrow Keys / WASD).
3. Synchronized frames and steering angles are stored in `dataset/images/` and `dataset/frame_*.json`.

### Phase 3: Train Behavioral Cloning Models
- **Discrete Classification:**
  ```bash
  python 3_train.py
  ```
- **Continuous Steering Angle Regression:**
  ```bash
  python 3_train_reg.py
  ```
Trained weights will be saved into the `models/` folder.

### Phase 4: Run Autonomous End-to-End Driving
1. In `4_run_model.py`, configure `MODEL_ID = "model1.h5"` (or `-1` for latest).
2. Set `4_run_model.py` as the car's controller in Webots and press **Play**.
3. The CNN will take live camera frames and autonomously steer the car around the city.

### Phase 5: LiDAR Collision Avoidance
1. Open Webots with `worlds/city.wbt`.
2. Attach `4_collision_avoidance/2_lidar_automated.py` as the vehicle controller.
3. The vehicle will project 2D LiDAR ranges into a Bird's-Eye View grid and autonomously steer away from obstacles.

### Phase 6: Explainable AI Verification
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook 5_explain.ipynb
   ```
2. Run all cells to inspect LIME superpixel saliency overlays and confirm the visual regions driving steering decisions.

---

## 📊 Performance Benchmarks & Results

### Road Sign Classification Comparison
| Metric | Classical ML (HOG + RBF SVM) | Deep Learning (CNN + SMOTE) |
|---|---|---|
| **Overall Accuracy** | **98.55%** | **99.26%** |
| **Speed Limit (Precision / Recall)** | 1.00 / 1.00 | 0.99 / 1.00 |
| **Stop Sign (Precision / Recall)** | 1.00 / 0.93 | 0.98 / 0.97 |
| **Traffic Light (Precision / Recall)** | 0.92 / 1.00 | 0.99 / 0.98 |
| **Crosswalk (Precision / Recall)** | 1.00 / 0.93 | 0.97 / 0.99 |
| **Inference Latency** | Low CPU footprint | Fast parallel GPU inference |

### Behavioral Cloning Autonomous Driving
- **Continuous Steering Angle MSE:** $< 0.021$ on validation holdout.
- **Latency:** Real-time inference executed within Webots basic simulation step ($< 16\text{ms}$ per control cycle).
- **Safety Envelope:** LiDAR collision avoidance successfully detects obstacles within an 80m range and executes collision-free corridor navigation.

---

## 🎤 How to Pitch & Describe This Project

### 📌 30-Second Elevator Pitch
> *"I built a full-stack autonomous vehicle system in the Webots robotics simulator featuring end-to-end perception, behavioral cloning, and LiDAR-based collision avoidance. The car uses classical computer vision for lane following, a deep CNN for camera-based steering control, and 2D LiDAR point clouds projected to a Bird's-Eye View grid for obstacle evasion. I also integrated SMOTE to solve class imbalance in road sign detection and used LIME Explainable AI to visually verify the neural network's decision-making process."*

### 💼 Resume Bullet Points
- **Developed a full-stack autonomous driving prototype** in Webots 3D simulator featuring camera-based lane keeping, end-to-end CNN steering prediction, and 2D LiDAR collision avoidance.
- **Trained behavioral cloning models (TensorFlow/Keras)** on human teleoperation data, achieving real-time autonomous navigation with sub-20ms inference latency.
- **Engineered a 2D LiDAR processing pipeline** converting 180° polar range measurements into Bird's-Eye View Cartesian occupancy grids for dynamic corridor steering.
- **Designed a robust road sign perception engine**, benchmarking HOG + SVM against Deep CNNs and resolving a 7:1 class imbalance using SMOTE to achieve **99.2% accuracy**.
- **Implemented Explainable AI (LIME)** to extract superpixel saliency maps, validating that model steering predictions correspond to genuine road boundary features.

### 🎯 Key Interview Questions & Answers
- **Q: Why use Behavioral Cloning instead of pure Computer Vision?**
  *A: Classical CV lane-centering relies on rigid assumptions (e.g. visible, high-contrast yellow lines). Behavioral cloning learns implicit road cues from human demonstrations—including road curvature, unmarked edges, and intersections—allowing generalized navigation where simple thresholding fails.*
- **Q: How did you tackle class imbalance in road sign classification?**
  *A: The dataset was heavily skewed toward speed limit signs (~75%). To prevent majority-class bias, I used SMOTE (Synthetic Minority Over-sampling Technique) in feature space to synthesize minority class samples, combined with stratified sampling and early stopping.*
- **Q: How does the LiDAR collision avoidance algorithm work?**
  *A: The SICK LMS 291 sensor provides a 1D array of 180 ranges across 180°. I transform these polar coordinates to vehicle-centric Cartesian coordinates, project them into an occupancy matrix, and scan a safety horizon row (`ROW_HIT = 32`). The algorithm computes the midpoint between adjacent obstacle clusters to steer toward the widest open corridor.*
- **Q: What role does LIME play in your system?**
  *A: Deep learning models in autonomous driving operate as black boxes. LIME perturbs superpixels to evaluate which visual features contribute positively or negatively to the steering prediction, ensuring the model focuses on lanes and road borders rather than background noise.*

---

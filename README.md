# Crowd Monitoring: Detection, Counting & Tracking

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![DeepSORT](https://img.shields.io/badge/Tracking-DeepSORT-orange)
![Trajectory](https://img.shields.io/badge/Trajectory-Visualization-red)

## 📜 Project Abstract
Accurate crowd detection and tracking are vital for public safety, event management, and disaster response. This project implements a deep learning-based pipeline designed to handle dense scenes, occlusion, and scale variations.

We integrate **YOLOv8 with SAHI** (Slicing Aided Hyper Inference) to improve small object detection and **DeepSORT** for robust multi-object tracking and **trajectory visualization**. We also experimented with **RT-DETR** (Real-Time Detection Transformer) for comparison.

## 📥 Project Presentation & Visuals (Deep Dive)
For a comprehensive look at the **visual outputs**, detection comparisons (YOLO vs SAHI), trajectory tracking results, and detailed architecture, please refer to our Deep Dive presentation:

👉 [**Deep Dive into Code, Architecture and Outputs**](./Capstone%20Phase2.pptx)

## 🚀 Key Features
* **Advanced Detection:** Utilizes YOLOv8 enhanced with SAHI to detect small, occluded individuals in dense crowds.
* **Trajectory Tracking:** Implements DeepSORT with Person Re-Identification (Re-ID) to visualize movement paths and maintain consistent IDs across frames.
* **Transformer Integration:** Comparative analysis using RT-DETR.
* **Custom Dataset:** Trained on a custom dataset annotated via MakeSenseAI, featuring Mosaic and MixUp augmentation.

## 📊 Performance Metrics
Our model achieves scalable tracking in real-time conditions.

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Precision** | **81%** | High accuracy in avoiding false positives |
| **Recall** | **61%** | Effectiveness in detecting objects |
| **mAP@0.5** | **0.71** | Mean Average Precision at 0.5 IoU |
| **IDF1** | **75.3%** | Identity assignment accuracy (Tracking) |
| **MOTA** | **68.1%** | Multi-Object Tracking Accuracy |

## 🛠️ Installation

Follow these steps to set up the environment and run the codebase.

### 1. Clone the Repository
```bash
git clone [https://github.com/krishi18/Crowd-Monitoring-Detection-Counting-Tracking.git](https://github.com/krishi18/Crowd-Monitoring-Detection-Counting-Tracking.git)
cd Crowd-Monitoring-Detection-Counting-Tracking
```

### 2. Create a Virtual Environment (Recommended)
It is best practice to use a virtual environment to avoid conflicts with other projects.

**For Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Weights
The model requires pre-trained weights to function.

- Create a folder named weights in the root directory of the project.

- Download the DeepSORT feature extraction model: [**Download Here**](https://drive.google.com/drive/folders/18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp)

- Place the downloaded file inside the weights/ folder. (Note: If you have your own trained YOLOv8 weights like best.pt, place them here as well).

## Usage:
To run the detection and tracking on a video file:

```bash
python track.py --source video.mp4 --yolo-weights weights/best.pt --tracker deep_sort --save-trajectory

# 👁️ GOD's EYE — Urban Intelligence System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-Desktop-41CD52?style=for-the-badge&logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Inference-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge)](LICENSE.txt)

**GOD's EYE** is a high-performance, Indian-made Urban Intelligence and Surveillance system designed for smart city command centers. It utilizes state-of-the-art Computer Vision and Predictive Analytics to monitor urban spaces, detect incidents in real-time, and forecast crowd risks.

---

## 🚀 Key Features

### 📡 Real-time Multi-Stream Monitoring
- **Heterogeneous Support**: Seamlessly monitor RTSP, HTTP, HLS (EarthCam/YouTube), and local camera feeds.
- **Auto-Healing Pipeline**: Intelligent reconnection logic with adaptive quality degradation for low-bandwidth environments.
- **Hardware Agnostic**: Automatic switching between **GPU (CUDA)** for maximum precision and **CPU (ONNX)** for ultra-lightweight deployment (<100MB RAM).

### 🧠 Advanced Computer Vision
- **Incident Detection**: Real-time detection of fire, smoke, vehicle accidents, and person falls.
- **Crowd Analytics**: Monitoring of people/vehicle density, movement speed, and direction entropy.
- **Risk Scoring**: Dynamic weighted risk assessment using normalized urban metrics.
- **Person Re-ID**: Tracking specific targets across multiple camera streams using deep feature embedding.

### 🔮 Predictive Intelligence
- **Hybrid Forecasting**: Utilizes LSTM (Long Short-Term Memory) and ARIMA models to predict crowd risks 5–15 minutes into the future.
- **Resource Recommender**: Automated deployment suggestions based on detected and predicted incidents.

---

## 🛠️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **Core UI** | PyQt5 (Python) |
| **Video Processing** | OpenCV, FFMPEG |
| **Deep Learning** | YOLOv8 (Ultralytics), ONNX Runtime |
| **Predictive Modeling** | PyTorch (LSTM), Statsmodels (ARIMA) |
| **Database** | SQLite (WAL mode for high concurrency) |
| **Web Extraction** | yt-dlp, Streamlink |

---

## 📦 Installation

> [!IMPORTANT]
> This is a proprietary system. Ensure you have the necessary dependencies installed for your hardware.

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "GOD's EYE"
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Application**:
   ```bash
   python main.py
   ```

---

## ⚖️ Legal & Licensing

**GOD's EYE is Proprietary Software.**

Copyright © 2026 **Kanishk Yadav**. All Rights Reserved.

- **Strict Limitation**: No part of this Software may be used, copied, modified, or distributed without the **express written consent** of Kanishk Yadav.
- **Unauthorized Usage**: Any unauthorized access or reproduction of this codebase will be met with legal action.

For inquiries or usage permissions, contact **Kanishk Yadav**.

---

## 🏗️ Architecture

The system follows a high-performance **Multiprocessing Architecture** to bypass the Python GIL:
- Each camera operates in its own OS thread/process.
- IPC (Inter-Process Communication) via compressed JPEG queues to minimize RAM consumption.
- Thread-local database connections to eliminate SQLite lock contention.

---
*Developed for the future of Indian Smart Cities.*

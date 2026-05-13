# Smart City Command Center
## User Manual — Enterprise Edition
### Indian Urban Intelligence Platform

---

**Author:** Kanishk Yadav  
**Version:** 2.0 — Production Release  
**Classification:** Operator Reference Document  
**Platform:** Windows Desktop (x64) | CPU-Optimised | 8GB RAM Minimum

---

> *This software is developed for deployment in Indian Smart City projects under the Ministry of Housing and Urban Affairs (MoHUA) Smart Cities Mission. It is designed for municipal control rooms, police command centres, and urban traffic management authorities.*

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [System Requirements](#2-system-requirements)
3. [Installation & First Launch](#3-installation--first-launch)
4. [Interface Layout](#4-interface-layout)
5. [Connecting Camera Feeds](#5-connecting-camera-feeds)
6. [Live Monitoring Operations](#6-live-monitoring-operations)
7. [Alerts & Incident Management](#7-alerts--incident-management)
8. [Person Tracking (Re-ID)](#8-person-tracking-re-id)
9. [Predictive Intelligence](#9-predictive-intelligence)
10. [Event Clips & Evidence](#10-event-clips--evidence)
11. [System Maintenance](#11-system-maintenance)
12. [RTSP Connection Reference](#12-rtsp-connection-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [Operator SOPs](#14-operator-sops)

---

## 1. System Overview

**Smart City Command Center** is a real-time AI-powered urban surveillance intelligence platform built for Indian municipal operations. It processes live video feeds from IP cameras, CCTV networks, and DVR/NVR systems to provide:

- **Real-time crowd density analysis** with risk scoring
- **Multi-class object detection** — persons, vehicles, bicycles, motorcycles, buses, trucks
- **Automated incident detection** — crowd surge, vehicle stoppage, fire detection
- **Predictive risk forecasting** using hybrid LSTM + ARIMA AI models
- **Person Re-Identification (Re-ID)** — track a specific individual across all camera feeds using a photograph
- **Human-in-the-loop decisions** — all AI recommendations require operator approval before action

The system operates **entirely offline**. No video, biometric, or surveillance data ever leaves the host machine.

---

## 2. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 (64-bit) | Windows 11 (64-bit) |
| RAM | 8 GB | 16 GB |
| CPU | Intel i5 / Ryzen 5 (6th gen+) | Intel i7 / Ryzen 7 |
| Storage | 50 GB free | 500 GB SSD |
| GPU | Not required | NVIDIA (optional, boosts FPS) |
| Display | 1366×768 | 1920×1080 or dual monitors |
| Network | LAN access to cameras | Gigabit LAN |
| Python | 3.9 – 3.11 | 3.10 |

---

## 3. Installation & First Launch

### Step 1 — Set Up Environment
```powershell
cd "E:\GOD's EYE"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Launch the Application
```powershell
python main.py
```

> The system will automatically create the required data directories (`data/events`, `data/models`, `data/logs`) on first launch.

### Step 3 — First-Time Configuration
Open `config.yaml` in any text editor to pre-configure your cameras:

```yaml
cameras:
  - id: "cam_01"
    name: "Main Gate — North"
    source: "rtsp://admin:pass@192.168.1.101:554/stream1"
  - id: "cam_02"
    name: "Market Square"
    source: "rtsp://admin:pass@192.168.1.102:554/stream1"
```

Cameras with a `source` value will auto-connect when the application starts.

---

## 4. Interface Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  HEADER: Smart City Command Center — [+ Add Camera Button]          │
├──────────────────────────────────────┬──────────────────────────────┤
│                                      │  ⚡ Alerts │ 📋 Log │ 🎬 Clips│
│     CAMERA TAB AREA                  │  🎯 Track  │ 🔮 Predict       │
│     (Live Video Feed / Connect Panel)│                              │
│                                      │  [Right Panel Content]       │
│                                      │                              │
├──────────────────────────────────────┴──────────────────────────────┤
│  STATS BAR: People: 0 | Vehicles: 0 | Density: 0.00 | Risk: 0.00  │
├─────────────────────────────────────────────────────────────────────┤
│  STATUS BAR: Status: Idle                              FPS: --      │
└─────────────────────────────────────────────────────────────────────┘
```

### Panel Descriptions

| Panel | Tab | Function |
|-------|-----|----------|
| Camera Area | Left side | Live video feeds, HUD overlay with risk/density data |
| Alerts | ⚡ Alerts | Real-time incident alerts with severity colour coding |
| Recommendations | ⚡ Alerts | AI-suggested deployment actions (requires operator approval) |
| Event Log | 📋 Log | Full chronological log of all detected incidents |
| Clip Viewer | 🎬 Clips | Playback of auto-saved event video clips |
| Target Tracking | 🎯 Track | Upload a photo to search for a person across all feeds |
| Prediction | 🔮 Predict | 15-minute ahead risk forecasting with AI confidence score |

---

## 5. Connecting Camera Feeds

### 5.1 Connecting an IP Camera / CCTV via RTSP

1. Click any camera tab (e.g., **Camera 1**)
2. The connect panel will be displayed with an RTSP URL input field
3. Enter the RTSP URL of your camera:
   ```
   rtsp://admin:yourpassword@192.168.1.x:554/stream1
   ```
4. Click **🔗 Connect RTSP**
5. The tab will switch to the live video feed within 3–5 seconds

### 5.2 Connecting a USB / Built-in Webcam

1. Click any camera tab
2. Click **📷 Open Webcam (Index 0)**
3. The system will open the default webcam at index 0

> To use a second webcam, add a new camera tab via **`+ Add Camera`** and enter `1` as the source.

### 5.3 Testing with a Video File

1. Click any camera tab
2. Click **📁 Open Video File**
3. Browse to your `.mp4`, `.avi`, `.mkv`, or `.mov` file
4. The system will process the file as if it were a live feed

### 5.4 Adding a New Camera

Click the **`+ Add Camera`** button in the top-right of the header. A new camera tab will appear. You can connect up to as many cameras as your hardware supports.

### 5.5 Removing a Camera

Click the **✕** on any camera tab to disconnect the feed and remove the tab.

---

## 6. Live Monitoring Operations

### 6.1 Understanding the Video HUD Overlay

Once a feed is connected, a heads-up display (HUD) is drawn on each video frame:

```
┌─────────────────────────────────────────────────────┐
│ RISK: 0.73 [HIGH]    People: 45  Vehicles: 8    cam_01 │
│                                                       │
│   [ Live video with bounding boxes on persons         │
│     and vehicles. Persons shown in green boxes,       │
│     vehicles in orange. Fire regions in red. ]        │
└─────────────────────────────────────────────────────┘
```

| HUD Element | Meaning |
|-------------|---------|
| `RISK: 0.73 [HIGH]` | Current crowd risk score (0.0 = safe, 1.0 = critical) |
| `People: 45` | Number of persons detected in current frame |
| `Vehicles: 8` | Number of vehicles detected |
| `cam_01` | Camera ID for this feed |

### 6.2 Risk Level Colour Coding

| Risk Level | Score Range | Indicator Colour | Recommended Action |
|------------|-------------|------------------|--------------------|
| LOW | 0.00 – 0.40 | 🟢 Green | Normal monitoring |
| MEDIUM | 0.40 – 0.70 | 🟡 Amber | Alert standby units |
| HIGH | 0.70 – 1.00 | 🔴 Red | Deploy response units |

### 6.3 Stats Bar

The stats bar at the bottom of the camera area shows real-time data for the currently active camera:

```
[cam_01]  People: 23 | Vehicles: 5 | Density: 0.61 | Speed: 12.3 | Entropy: 0.84
```

- **Density**: Normalised crowd density (0.0–1.0)
- **Speed**: Average movement speed of tracked objects (pixels/frame)
- **Entropy**: Direction entropy — high entropy means chaotic, unpredictable crowd movement

---

## 7. Alerts & Incident Management

### 7.1 Alert Types

The system automatically detects and raises alerts for the following events:

| Event Type | Trigger Condition | Default Severity |
|------------|-------------------|-----------------|
| Crowd Surge | Risk score > 0.70 | HIGH |
| Crowd Buildup | Risk score 0.40–0.70 | MEDIUM |
| Vehicle Stoppage | Vehicle stopped in a moving zone | MEDIUM |
| Fire Detected | HSV colour signature of fire | CRITICAL |
| Fall Detected | Person's aspect ratio drops sharply | HIGH |
| Prediction Pre-Alert | AI forecast > 0.70 for next 15 min | HIGH |

### 7.2 Responding to an Alert

When an alert is raised:

1. The **⚡ Alerts** tab will automatically come to the foreground
2. A new alert card appears with: event type, camera ID, timestamp, severity, and confidence
3. The **Recommended Action** panel below shows the AI's suggested deployment
4. Review the recommendation and choose one of:
   - ✅ **CONFIRM DEPLOYMENT** — Approve the action (logged as operator-confirmed)
   - ❌ **REJECT** — Dismiss the recommendation (logged as operator-rejected)

> **All decisions are logged.** Neither confirming nor rejecting triggers any external action — it is purely a decision-support and audit trail system.

### 7.3 Severity Colour Reference

| Colour | Severity | Description |
|--------|----------|-------------|
| 🔴 Dark Red | CRITICAL | Immediate threat — fire, mass casualty risk |
| 🟠 Amber | HIGH | Serious incident requiring rapid response |
| 🟡 Yellow | MEDIUM | Developing situation, monitor closely |
| 🟢 Green | LOW | Informational, no action required |

---

## 8. Person Tracking (Re-ID)

The **🎯 Track** tab allows operators to upload a photograph of a specific individual and track that person across all connected camera feeds in real-time.

### 8.1 How It Works

The system uses a MobileNetV2 neural network to extract a 1280-dimensional mathematical "signature" from the uploaded photo. It then compares this signature against every person bounding box detected in every camera frame. When the similarity exceeds the set threshold, a match alert is raised.

### 8.2 Step-by-Step: Tracking a Person

**Step 1:** Click the **🎯 Track** tab in the right panel

**Step 2:** Click **📁 Browse Photo**
- Select a clear photograph of the person
- Best results: frontal or 3/4 view, full body, good lighting
- Acceptable formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

**Step 3:** Adjust **Match Sensitivity**
- **50–60%**: Broad match — catches more candidates, more false positives
- **70%** (default): Balanced — recommended for most operations
- **85–95%**: Strict match — high confidence, may miss the person in poor lighting

**Step 4:** Click **🎯 Start Tracking**

The system will immediately begin scanning every frame from every connected camera. When the person is spotted, the target card updates:

```
🚨 SPOTTED — cam_03  (87% match)
Last seen: 14:32:07
```

### 8.3 Best Practices for Re-ID Photos

- Use the **clearest available photograph** — CCTV still frame, ID card scan, or mobile photo
- The photo should show the person's **clothing, not just their face** (the algorithm matches appearance, not facial features — by design, for privacy compliance)
- If the person has changed clothes, upload a new photo
- Upload **multiple photos** of the same person at different angles for better tracking accuracy

### 8.4 Removing a Target

Click the **✕** button on any target card to stop tracking that person.

---

## 9. Predictive Intelligence

### 9.1 Overview

The **🔮 Predict** tab shows AI-generated risk forecasts for the next 15 minutes, based on the last 200 frames of crowd behaviour data. The system uses a hybrid model:

- **LSTM Neural Network (60% weight)**: Detects long-term temporal patterns in crowd movement
- **ARIMA Statistical Model (40% weight)**: Detects cyclical and trend-based patterns

### 9.2 Reading the Prediction Panel

```
Predicted Risk (15 min): 0.81
Risk Level: HIGH
Recommended Action: Pre-deploy police and emergency units. Maximum alert.

LSTM Component: 0.79
ARIMA Component: 0.84
```

### 9.3 Training the Prediction Models

For best results, train the models on your city's historical crowd data:

1. Go to **File → 🧠 Train Prediction Models**
2. The system will train on the last 200+ logged data points
3. Training takes approximately 2–5 minutes on a standard laptop
4. Models are saved to `data/models/` for future sessions

> First-time use: Go to **View → 📊 Generate Synthetic Data** to create training data before the system has collected real-world logs.

### 9.4 Prediction Update Interval

The prediction model re-runs every 60 seconds (configurable in `config.yaml` under `prediction.update_interval_seconds`).

---

## 10. Event Clips & Evidence

### 10.1 Automatic Clip Saving

When an incident is detected, the system automatically saves the last **20 seconds** of video as an MP4 clip to `data/events/`. File naming format:

```
event_20260506_143207_crowd_surge.mp4
```

### 10.2 Viewing a Clip

**Method 1 — From Event Log:**
1. Click the **📋 Log** tab
2. Find the event in the table
3. Double-click the row (or click the clip icon) to open it in the Clip Viewer

**Method 2 — Direct:**
1. Click the **🎬 Clips** tab
2. The clip viewer will show the last loaded clip
3. Use **▶ Play**, **⏸ Pause**, and **⏹ Stop** to control playback

### 10.3 Evidence Retention Policy

By default, the system automatically deletes event clips and database records **older than 30 days** each time the application starts. To change this:

Edit `ui/main_window.py`, line:
```python
self.db.prune_old_data(days=30)   # Change 30 to your required retention period
```

> **Legal note:** Ensure your retention period complies with applicable Indian data protection regulations and local municipal guidelines.

---

## 11. System Maintenance

### 11.1 Daily Checklist (Operator)

- [ ] Verify all camera feeds are connected and showing live video
- [ ] Confirm FPS reading in the status bar is above 4.0
- [ ] Check `data/events/` folder size is not approaching disk capacity
- [ ] Review previous session's event log for any unresolved HIGH/CRITICAL alerts
- [ ] Confirm prediction model is running (check 🔮 Predict tab)

### 11.2 Weekly Checklist (System Administrator)

- [ ] Re-train prediction models with accumulated real-world data (File → Train Models)
- [ ] Review and archive important event clips before automatic pruning
- [ ] Check Windows Event Viewer for hardware errors
- [ ] Verify network connectivity to all camera IP addresses
- [ ] Check available disk space — keep at least 20GB free

### 11.3 Data Directory Structure

```
data/
├── events/          ← Auto-saved incident video clips (MP4)
├── models/          ← Trained AI model weights (DO NOT DELETE)
│   ├── lstm_model.pt
│   └── arima_model.pkl
├── logs/            ← CSV training data and synthetic data
└── gods_eye.db      ← SQLite database (all events, predictions, logs)
```

---

## 12. RTSP Connection Reference

### 12.1 Common RTSP URL Formats by Brand

| Camera Brand | RTSP URL Format |
|-------------|-----------------|
| **Hikvision** | `rtsp://admin:PASSWORD@IP:554/Streaming/Channels/101` |
| **Hikvision (Sub-stream)** | `rtsp://admin:PASSWORD@IP:554/Streaming/Channels/102` |
| **Dahua** | `rtsp://admin:PASSWORD@IP:554/cam/realmonitor?channel=1&subtype=0` |
| **CP Plus** | `rtsp://admin:PASSWORD@IP:554/cam/realmonitor?channel=1&subtype=0` |
| **Axis** | `rtsp://admin:PASSWORD@IP/axis-media/media.amp` |
| **Reolink** | `rtsp://admin:PASSWORD@IP:554/h264Preview_01_main` |
| **Amcrest** | `rtsp://admin:PASSWORD@IP:554/cam/realmonitor?channel=1` |
| **Bosch** | `rtsp://IP/rtsp_tunnel` |
| **Generic ONVIF** | `rtsp://admin:PASSWORD@IP:554/stream1` |
| **Generic (Alt)** | `rtsp://admin:PASSWORD@IP:554/live` |

### 12.2 Finding Your Camera's RTSP URL

1. Log into your NVR/DVR web interface
2. Go to **Network Settings → Advanced → RTSP**
3. Note the port (default: 554) and URL pattern
4. Use **ONVIF Device Manager** (free software) to auto-discover cameras on your LAN

### 12.3 Network Configuration Requirements

- Camera and the PC running this software must be on the **same LAN** or connected via VPN
- Ensure port **554** (RTSP) is not blocked by Windows Firewall
- For remote access, configure port forwarding on your router

---

## 13. Troubleshooting

### Problem: Camera tab shows "Waiting for video feed" and does not connect

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| Wrong RTSP URL | Verify URL format for your camera brand (see Section 12) |
| Wrong credentials | Check username/password in the RTSP URL |
| Camera offline | Ping the camera IP: `ping 192.168.1.x` in PowerShell |
| Firewall blocking | Allow Python through Windows Firewall |
| Wrong port | Try port 8554 or 80 instead of 554 |

### Problem: FPS shows below 2.0

- Reduce detection confidence in `config.yaml`: `confidence: 0.50` (higher = fewer detections = faster)
- Close other applications consuming CPU
- Reduce the number of simultaneously active camera feeds
- Lower resolution in config: set `width: 640, height: 480`

### Problem: Application crashes on startup

Run from the activated virtual environment:
```powershell
venv\Scripts\activate
python main.py
```
Check the console output for the specific error message.

### Problem: Re-ID / Track tab shows "Could not load tracking model"

Install the required library:
```powershell
pip install torchvision
```

### Problem: Event clips not being saved

- Check that `data/events/` directory exists and has write permissions
- Ensure at least 2GB of free disk space is available

---

## 14. Operator SOPs

### SOP-01: Responding to a CRITICAL Alert (Fire / Mass Crowd)

1. **Acknowledge** the alert in the ⚡ Alerts tab immediately
2. **Switch** to the camera tab for the affected location
3. **Visually confirm** the incident on the live feed
4. **Review** the AI Recommended Action in the panel
5. **Contact** the relevant emergency service (fire brigade / police PCR)
6. **Click CONFIRM DEPLOYMENT** to log the decision
7. Continue monitoring the feed until the situation is resolved
8. **Document** the outcome in the operator note field

### SOP-02: Person Search Operation

1. Obtain the clearest available photograph of the target person
2. Navigate to the **🎯 Track** tab
3. Click **Browse Photo** and select the photograph
4. Set sensitivity to **75%** for urban operations (adjust based on results)
5. Click **Start Tracking**
6. Monitor all camera feeds — a match alert will appear on the target card
7. Note the camera ID and timestamp when the person is spotted
8. Relay location information to field units
9. Remove the target card when the operation concludes

### SOP-03: Shift Handover

1. Review the **📋 Log** tab — brief the incoming operator on any unresolved HIGH/CRITICAL events
2. Confirm all cameras are connected and streaming
3. Do not close the application during handover — incoming operator should assume the running session
4. Log the handover time in the operator's physical logbook

---

## Appendix A: Keyboard Quick Reference

| Action | Method |
|--------|--------|
| Add new camera | Click `+ Add Camera` button |
| Remove camera | Click ✕ on tab |
| Switch to Alerts | Click ⚡ Alerts tab |
| Switch to Tracking | Click 🎯 Track tab |
| Confirm alert action | Click ✅ CONFIRM DEPLOYMENT |
| Reject alert action | Click ❌ REJECT |

---

## Appendix B: Configuration Quick Reference (`config.yaml`)

| Parameter | Location | Effect |
|-----------|----------|--------|
| `detection.confidence` | `config.yaml` | Lower = more detections, slower; Higher = fewer, faster |
| `video.target_fps` | `config.yaml` | Processing frame rate (default: 8 FPS) |
| `risk.thresholds.high` | `config.yaml` | Score above which HIGH alert is triggered (default: 0.70) |
| `buffer.duration_seconds` | `config.yaml` | Length of saved event clips (default: 20 seconds) |
| `prediction.update_interval_seconds` | `config.yaml` | How often the AI re-runs predictions (default: 60 seconds) |

---

## Document Information

| Field | Value |
|-------|-------|
| **Author** | Kanishk Yadav |
| **Document Version** | 2.0 |
| **Software Version** | Smart City Command Center v2.0 |
| **Last Updated** | May 2026 |
| **Classification** | Operator Reference — Internal Use |
| **Intended Audience** | Municipal Control Room Operators, Security Supervisors, System Administrators |

---

*Smart City Command Center is developed for Indian Smart City deployments. All AI decisions are advisory only and require human operator confirmation. The system does not perform facial recognition — Re-ID is based on clothing and physical appearance features only, in compliance with privacy-by-design principles.*

*For technical support, contact the system administrator.*

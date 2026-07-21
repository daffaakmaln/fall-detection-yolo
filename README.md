# SentinelAI — Hybrid Fall Detection (Python)

```
╔════════════════════════════════════════════════════════════════╗
║  ____  _____ _   _ _____ ___ _   _ _____ _          _    ___    ║
║ / ___|| ____| \ | |_   _|_ _| \ | | ____| |        / \  |_ _|   ║
║ \___ \|  _| |  \| | | |  | ||  \| |  _| | |       / _ \  | |    ║
║  ___) | |___| |\  | | |  | || |\  | |___| |___   / ___ \ | |    ║
║ |____/|_____|_| \_| |_| |___|_| \_|_____|_____| /_/   \_\___|   ║
╚════════════════════════════════════════════════════════════════╝
```

 See Our FullStack Repo here:
- **Fall Detection AI**: [fall-detection-yolo](https://github.com/daffaakmaln/fall-detection-yolo) (Python)
- **Backend**: [backend-sentinel](https://github.com/JonathanFaustinus/backend-sentinel) (Javascript)
- **Frontend**: [sentinel_ai_app](https://github.com/daffaakmaln/sentinel_ai_app) (Flutter - Dart)

A real-time fall detection system based on **YOLOv8 Pose Estimation** + **Random Forest Classifier**, integrated with a **Backend API** (Node.js), **Telegram Bot** for flexibility command, and **Flutter** for the Mobile Application

---

## Table of Contents

- [Key Features](#-key-features)
- [Architecture & How It Works](#-architecture--how-it-works)
- [Fall Detection State Machine](#-fall-detection-state-machine)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Environment Variables](#-environment-variables)
- [Running the Program](#-running-the-program)
- [Telegram Bot Commands](#-telegram-bot-commands)
- [Backend API Integration](#-backend-api-integration)
- [Parameter Tuning](#-parameter-tuning)
- [GUI / Debug Overlay](#-gui--debug-overlay)
- [Troubleshooting](#-troubleshooting)

---

##  Key Features

- **Real-time pose detection** using YOLOv8 (`yolov8s-pose.pt`)
- **Hybrid classification**: combines body angle + fall speed, processed by a **Random Forest** model (`model_fall_detection.pkl`)
- **EMA smoothing** and **median filtering** to reduce noise in position/speed estimation
- **3-stage state machine** (SAFE → SUSPECT_FALL → FALL_CONFIRMED) to avoid momentary false positives
- **Voting mechanism** to validate impact before classifying it as a fall
- **Auto-recovery**: status automatically returns to safe if the person stands back up within a set duration
- **Telegram notifications** (text + photo) when an incident is detected
- **Bot command polling** (`/start`, `/status`, `/reset`, `/help`)
- **Backend integration**: periodically sends fall events & camera status frames via REST API
- **Threading** for all HTTP requests so the main camera loop is never blocked

---

## Architecture & How It Works

```
Camera (cv2.VideoCapture)
        │
        ▼
YOLOv8 Pose Estimation  →  shoulder & hip keypoints
        │
        ▼
hitung_fitur() → body angle + posisi_y (EMA smoothed)
        │
        ▼
Speed calculation (position delta / time delta) → median filter
        │
        ▼
Random Forest Model → prediction label (fall / normal) + confidence
        │
        ▼
State Machine (SAFE / SUSPECT_FALL / FALL_CONFIRMED)
        │
        ├──► Telegram Notification (when notification threshold is hit)
        └──► Backend API (POST /api/events/fall)
```

### Features extracted from keypoints

Uses the **shoulder (index 5, 6)** and **hip (index 11, 12)** keypoints from YOLOv8 pose output:

| Feature | Description |
|---|---|
| `sudut` (angle) | Tilt angle of the body (shoulder–hip line) relative to the vertical axis, in degrees |
| `posisi_y` (y-position) | Vertical midpoint between shoulders and hips, smoothed with EMA (`alpha=0.7`) |
| `kecepatan` (speed) | Change in `posisi_y` per second (estimated fall speed) |
| `kecepatan_filtered` | Median of the last 5 speed samples |
| `kecepatan_puncak` (peak speed) | Maximum speed value within the last 30-frame window |

If shoulder/hip keypoint confidence is below `0.5`, that frame is **ignored** (treated as invalid data).

---

## Fall Detection State Machine

| State | Entry Trigger | Exit Trigger |
|---|---|---|
| **SAFE** | Default / after recovery | `impact_terdeteksi = True` → moves to `SUSPECT_FALL` |
| **SUSPECT_FALL** | Impact detected (peak speed + angle + RF voting) | `confirm_hits >= POST_FALL_REQUIRED_FRAMES` → moves to `FALL_CONFIRMED` |
| **FALL_CONFIRMED** | Post-fall condition consistently confirmed | Angle < `RECOVER_ANGLE_THRESHOLD` for `RECOVER_TIME_SEC` seconds → returns to `SAFE` |

### `impact_terdeteksi` (impact detected) condition
```python
kecepatan_puncak > IMPACT_SPEED_THRESHOLD
and sudut > IMPACT_ANGLE_THRESHOLD
and impact_vote_ok   # >= 2 out of the last 5 RF predictions = "fall"
```

### `post_fall_condition`
```python
sudut > POST_FALL_ANGLE_THRESHOLD
OR posisi_smooth > (baseline_posisi_y + POST_FALL_Y_OFFSET)
```

`baseline_posisi_y` is automatically computed from the average position of the person while standing normally (SAFE state, angle < 45°), updated via EMA (`0.9` old / `0.1` new).

---

## Requirements

```
opencv-python
ultralytics
numpy
joblib
requests
python-dotenv
```

Plus the model files:
- `yolov8s-pose.pt` — YOLOv8 pose estimation model (auto-downloaded by Ultralytics if not present)
- `model_fall_detection.pkl` — your own trained Random Forest model (must be provided manually in the same folder)

---

## Installation

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install opencv-python ultralytics numpy joblib requests python-dotenv

# 3. Make sure the model files exist in the root folder:
#    - yolov8s-pose.pt
#    - model_fall_detection.pkl

# 4. Create a .env file (see the Environment Variables section)
```

---

## Configuration

Main configuration values are at the top of the script:

```python
BACKEND_URL       = "http://localhost:3000"   # Node.js backend URL
CAMERA_ID         = 2                         # Logical camera ID registered in the backend
JEDA_KIRIM_FRAME  = 5                         # interval (seconds) for sending status frames to backend
JEDA_NOTIFIKASI   = 30                        # delay between notifications (seconds)
WAKTU_DIAM        = 1.5                       # (reserved, not yet actively used in the loop)
```

>  `kamera = cv2.VideoCapture(0)` is still hardcoded to device `0`, separate from the `CAMERA_ID` variable sent to the backend (which is only the logical camera ID in the database, not the device index).

---

## Environment Variables

Create a `.env` file in the root folder:

```env
API_KEY=your_backend_api_key
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

| Variable | Purpose |
|---|---|
| `API_KEY` | Sent as the `x-api-key` header on requests to the backend |
| `TELEGRAM_TOKEN` | Telegram bot token (from @BotFather) |
| `TELEGRAM_CHAT_ID` | Destination chat ID for notifications (the bot only responds to this chat ID) |

---

## Running the Program

```bash
python sentinel.py
```

- Press **`Q`** in the video window to exit the program.
- Make sure the Node.js backend is running at `BACKEND_URL` before starting (if not, requests will fail, but the error is only logged to the console — detection keeps running).
- If testing with a physical device (e.g. phone/Android) against a local backend, use ADB reverse tunneling:
  ```bash
  adb reverse tcp:3000 tcp:3000
  ```

---

## Telegram Bot Commands

The bot polls for updates every 2 seconds (`cek_perintah()` runs on a separate thread).

| Command | Function |
|---|---|
| `/start` | Welcome message |
| `/status` | Shows the current status + sends a photo of the current frame |
| `/reset` | Resets all detection state back to `SAFE`, clears all history |
| `/help` | Shows the list of available commands |

> The bot only responds to messages from the configured `TELEGRAM_CHAT_ID`.

---

## Backend API Integration

### 1. Send Fall Event
```
POST {BACKEND_URL}/api/events/fall
Headers: x-api-key: <API_KEY>
Body (multipart/form-data):
  - camera_id
  - event_time
  - confidence_score (optional)
  - snapshot (jpg file)
```
Sent **once per incident** (controlled by the `fall_sudah_dikirim` flag, reset when state returns to `SAFE`).

### 2. Send Camera Status Frame
```
POST {BACKEND_URL}/api/cameras/{CAMERA_ID}/status-frame
Headers: x-api-key: <API_KEY>
Body (multipart/form-data):
  - frame (jpg file)
```
Sent periodically every `JEDA_KIRIM_FRAME` seconds (default 5 seconds), containing a snapshot of the current (annotated) frame.

All HTTP requests run on a **separate daemon thread** so they never block the main camera loop.

---

## Parameter Tuning

| Parameter | Default | Description |
|---|---|---|
| `RF_FALL_CONF_THRESHOLD` | `0.60` | Minimum RF model confidence for a prediction to count as "fall" |
| `IMPACT_SPEED_THRESHOLD` | `95` | Minimum peak speed (px/sec) indicating an impact |
| `IMPACT_ANGLE_THRESHOLD` | `40` | Minimum body angle at the moment of impact detection |
| `IMPACT_DELTA_ANGLE` | `10` | (reserved, calculated but not yet used as an active condition) |
| `IMPACT_WINDOW_SEC` | `0.6` | Maximum time window for computing angle delta |
| `IMPACT_VOTE_WINDOW` | `5` | Number of recent frames used for voting |
| `IMPACT_VOTE_REQUIRED` | `2` | Minimum number of "fall" votes from RF required to confirm impact |
| `KECEPATAN_PUNCAK_WINDOW` | `30` | Number of frames used to compute peak speed |
| `POST_FALL_CONFIRM_WINDOW` | `2.0` | (reserved, not yet directly used in the loop) |
| `POST_FALL_REQUIRED_FRAMES` | `6` | Number of post-fall hits (out of a max 30-frame window) needed to confirm FALL_CONFIRMED |
| `POST_FALL_ANGLE_THRESHOLD` | `65` | Minimum angle to count as a post-fall condition |
| `POST_FALL_Y_OFFSET` | `60` | Offset from baseline `posisi_y` to count as a post-fall condition |
| `RECOVER_ANGLE_THRESHOLD` | `35` | Maximum angle to count as starting to recover |
| `RECOVER_TIME_SEC` | `2.0` | Duration the recovery condition must hold before returning to SAFE |
| `EMA_ALPHA` | `0.7` | Smoothing weight for `posisi_y` (higher = slower to change) |

> Note: a few parameters (`IMPACT_DELTA_ANGLE`, `POST_FALL_CONFIRM_WINDOW`, `WAKTU_DIAM`) are already defined but not yet actively used in the logic — likely reserved for future tuning/development.

---

## GUI / Debug Overlay

The OpenCV window (`SentinelAI - Hybrid Fall Detection`) displays:

- Status (SAFE / SUSPECTED FALL / FALL DETECTED!) — shown in red when status contains "JATUH" (fall)
- Current body angle
- Raw speed & filtered speed
- Peak speed (30-frame window)
- Random Forest prediction result with confidence
- Impact voting score (`x/5`)
- Post-fall confirmation score (`x/6`)
- Pose skeleton/keypoint overlay (from `hasil[0].plot()`)

---

## Troubleshooting

| Issue | Likely Cause | Solution |
|---|---|---|
| Camera won't open | Wrong device index / already in use by another app | Change `cv2.VideoCapture(0)` to a different index |
| Error loading RF model | `model_fall_detection.pkl` not found / corrupted | Verify the model path and file integrity |
| Fall event not sent | Backend is down / wrong `API_KEY` / network blocked | Check `[Backend]` logs in the console, ensure backend and ADB reverse are running |
| Telegram bot not responding | Wrong token/chat ID | Check `.env`, make sure the chat ID matches the account being used |
| Too many false positives | Thresholds too sensitive | Increase `IMPACT_SPEED_THRESHOLD`, `IMPACT_ANGLE_THRESHOLD`, or `POST_FALL_REQUIRED_FRAMES` |
| Status doesn't return to safe | Person still in frame at a high angle | Make sure the angle stays below `RECOVER_ANGLE_THRESHOLD` consistently for `RECOVER_TIME_SEC` seconds |

---

##  Important Note

> **The core detection logic (pose extraction, feature engineering, state machine, and the RF model) must not be modified** without a thorough re-tuning and re-testing process, since all thresholds are interdependent.

# Driver Monitor with Eye Detection and Alarm

## Description
This project is a real-time driver drowsiness monitoring system that uses Mediapipe for eye detection and simpleaudio for alarm playback.  
It detects whether the eyes are open or closed, displays the status on screen, and plays an alarm (`alarm.wav`) if eyes remain closed too long (e.g 4-5 seconds).

This repository includes a **testbench folder** with all the necessary files to set up and run the project.

---

## Features
- Real-time webcam monitoring  
- Eye open/closed detection using Mediapipe Face Mesh  
- Alarm plays automatically when eyes are closed longer than 1 second  
- On-screen status text:
  - Green: Eyes OPEN  
  - Red: ALERT! Eyes CLOSED  
- Safe quit by pressing **`q`**  

---

## Files
- `driver_monitor_mediapipe.py` — Main Python script  
- `alarm.wav` — Audio file for alarm (must be in the same folder as the script)  
- `README.md` — Project documentation  

---

## Requirements
- Python 3.10 or 3.11 (Mediapipe does not support 3.13 on macOS yet)  
- Packages:
  ```bash
  python3.10 -m pip install mediapipe opencv-python simpleaudio

---

## Dependencies

You need **Python 3.10 or 3.11** (Mediapipe is not yet compatible with Python 3.13).

Install required Python packages:

```bash
python3.10 -m pip install --upgrade pip
python3.10 -m pip install mediapipe opencv-python simpleaudio

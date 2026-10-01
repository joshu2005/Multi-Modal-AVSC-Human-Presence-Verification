# Multi-Modal AVSC – Human Presence Verification

A multi-modal audio-video system designed for human presence verification using facial, behavioral, and audio-based cues.

## Project Overview

This project explores a multi-modal approach to human presence verification by combining multiple behavioral and facial cues. The system includes components for eye-blink detection, mouth movement analysis, head-pose estimation, and drowsiness detection.

The goal is to improve the reliability of human presence verification by analyzing multiple signals rather than relying on a single visual cue.

## Key Components

- Eye Aspect Ratio (EAR) based eye-blink detection
- Mouth Aspect Ratio (MAR) based mouth movement analysis
- Head-pose estimation
- Drowsiness detection
- Facial landmark-based analysis
- Audio-video based human presence verification

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Dlib
- NumPy
- Librosa
- PyAudio
- SciPy

## Project Files

```text
Multi-Modal-AVSC-Human-Presence-Verification/
│
├── app.py
├── drow_detection.py
├── EAR.py
├── HeadPose.py
├── MAR.py
├── .gitignore
├── README.md
└── requirements.txt
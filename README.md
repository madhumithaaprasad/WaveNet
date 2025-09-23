# WaveNet – Real-Time Hand Gesture Recognition System

## Project Description 
WaveNet is a real-time hand gesture recognition system that leverages computer vision and machine learning techniques to detect and classify hand gestures with high accuracy. Using **MediaPipe Hands**, the system extracts 21 3D hand landmarks from a live video feed, allowing precise gesture detection. **OpenCV** processes the webcam feed and provides visual feedback, while landmark-based geometric analysis differentiates gestures such as **peace sign, thumbs-up, OK, and wave**.

The system is designed for low-latency processing, ensuring smooth live interaction. It can be extended to control external applications using **PyAutoGUI**, enabling gesture-based system actions like slide navigation, volume adjustment, and cursor control. WaveNet serves as a foundation for **touchless interfaces, accessibility tools**, and **human-computer interaction (HCI)** experiments.

## Features
- Real-time hand detection using **MediaPipe Hands**
- Recognition of multiple gestures (peace, thumbs-up, OK, wave, etc.)
- Live webcam feed with on-screen visualization
- Low-latency pipeline optimized for smooth gesture tracking
- Optional integration with system actions via **PyAutoGUI**
- Foundation for **HCI applications** and **touchless controls**

## Tech Stack
- **Programming Language:** Python  
- **Computer Vision:** OpenCV  
- **Machine Learning / Landmark Detection:** MediaPipe Hands  
- **Visualization:** OpenCV Drawing Utilities, NumPy  
- **Optional Extension:** PyAutoGUI (gesture-controlled actions)

## Installation
1. Clone the repository:  
```bash
git clone https://github.com/yourusername/WaveNet.git
cd WaveNet

NextWave Gesture Controller (ML-Powered)
Project Overview
The NextWave Gesture Controller is a real-time computer vision application that uses a custom-trained Convolutional Neural Network (CNN) to recognize specific hand gestures captured by a webcam. It maps these recognized gestures to system controls (like media playback and volume), providing a novel, hands-free interface for desktop control.

This project demonstrates strong skills in machine learning lifecycle management (Data Collection, Training, and Deployment/Integration), computer vision, and systems programming.

Technical Highlights
Model Performance: Achieved a 98.36% Test Accuracy on a balanced, custom-collected dataset.

Dataset: Built a proprietary dataset of over 1,000 samples of hand landmarks across three distinct gestures (peace_sign, fist, open_hand).

Real-Time Processing: Utilizes Google's MediaPipe for fast, efficient hand landmark extraction and uses a multi-layer Keras model for low-latency prediction.

System Integration: Employs the pynput library to map predicted gestures to actual media control commands (Play/Pause, Volume Up/Down).

Gestures and Mapped Actions
Gesture

Mapped System Action

Description

peace_sign

Media Play/Pause

Toggles media playback in foreground applications.

fist

Volume Down (Scroll)

Decreases system volume via a simulated mouse scroll.

open_hand

Volume Up (Scroll)

Increases system volume via a simulated mouse scroll.

🚀 Getting Started
Prerequisites
Python 3.8+

A functional webcam.

Installation
Clone the repository and install all necessary dependencies using the provided requirements.txt file:

# Install dependencies (ensure you are using a virtual environment)
pip install -r requirements.txt

Running the Application
Ensure the trained model file (gesture_model.h5) and the data files (*.csv) are present in the project structure.

Execute the main application loop from the project's root directory:

python src/main.py

The webcam window will open, and recognized actions will be displayed in the console and at the top of the video feed. Press 'q' to quit the application.

Project Structure
NEXTWAVE/
├── data/                    # Custom collected landmark data files
│   ├── fist.csv
│   ├── open_hand.csv
│   └── peace_sign.csv
├── src/                     # Main application source code
│   ├── models/
│   │   ├── gesture_predictor.py  # Loads and runs the trained model
│   │   └── __init__.py
│   └── main.py              # Main execution file (Webcam capture, integration, control)
├── gesture_model.h5         # The final trained Keras/TensorFlow model (98.36% accurate)
├── data_collector.py        # Script used for raw data collection
├── train_model.py           # Script used for training and saving the model
├── requirements.txt         # List of all Python dependencies
└── README.md                # Project documentation

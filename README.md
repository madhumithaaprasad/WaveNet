# NextWave Gesture Controller (ML-Powered)

The **NextWave Gesture Controller** is a real-time computer vision application that leverages a custom-trained **Convolutional Neural Network (CNN)** to recognize hand gestures from a webcam and map them to system-level controls such as media playback and volume adjustment. This project demonstrates expertise in **machine learning lifecycle management, computer vision, and systems programming**.

---

## Project Overview
- **Objective**: Provide a novel, hands-free interface for desktop control using hand gestures.  
- **Key Skills Demonstrated**:
  - Machine Learning (data collection, training, deployment)
  - Real-time computer vision with MediaPipe
  - Systems integration with Python libraries
  - End-to-end model development and testing

---

## Technical Highlights
- **Model Performance**: Achieved **98.36% test accuracy** on a balanced, custom-collected dataset.  
- **Dataset**: Proprietary dataset of **1,000+ samples** across three gestures (`peace_sign`, `fist`, `open_hand`).  
- **Real-Time Processing**: Efficient landmark extraction using **Google MediaPipe** combined with a **Keras CNN** for low-latency prediction.  
- **System Integration**: Mapped gestures to desktop controls using the **pynput** library.  

---

## Gestures and Actions

| Gesture      | Mapped Action       | Description                                   |
|--------------|---------------------|-----------------------------------------------|
| ✌️ peace_sign | Media Play/Pause    | Toggles media playback in the active window.  |
| ✊ fist       | Volume Down         | Decreases system volume via simulated scroll. |
| 🖐️ open_hand  | Volume Up           | Increases system volume via simulated scroll. |

---

## Project Structure
```
NEXTWAVE/
├── data/ # Collected landmark data
│ ├── fist.csv
│ ├── open_hand.csv
│ └── peace_sign.csv
├── src/
│ ├── models/
│ │ ├── gesture_predictor.py # Loads and runs the trained CNN model
│ │ └── init.py
│ └── main.py # Main execution file (Webcam capture + system control)
├── gesture_model.h5 # Final trained Keras/TensorFlow model (98.36% accuracy)
├── data_collector.py # Script for raw data collection
├── train_model.py # Script for model training and saving
├── requirements.txt # List of Python dependencies
└── README.md # Project documentation

```

## Model Details
- **Frameworks**: TensorFlow/Keras, MediaPipe  
- **Architecture**: Multi-layer CNN trained on normalized hand landmark vectors  
- **Training Pipeline**:  
  1. Data collection with `data_collector.py`  
  2. Model training via `train_model.py`  
  3. Deployment and inference in `main.py`  
- **Accuracy**: 98.36% on test dataset  

---

## Future Enhancements
- Expand gesture set (mute, next/previous track, brightness control).
- Improve robustness under varying lighting and camera angles.  
- Cross-platform compatibility (Linux/macOS).  
- Integration with IoT devices for smart home control.  


---

## Getting Started

### Prerequisites
- Python **3.8+**
- Functional **webcam**
- Virtual environment recommended

### Installation
```bash
git clone https://github.com/your-username/nextwave-gesture-controller.git
cd nextwave-gesture-controller
pip install -r requirements.txt
```

<img width="787" height="629" alt="image" src="https://github.com/user-attachments/assets/6890d509-adea-4b5a-9107-7052a8ee4efe" />
<img width="792" height="636" alt="image" src="https://github.com/user-attachments/assets/8605e291-9289-4585-a7fb-b1a74fbe4bf7" />
<img width="788" height="629" alt="image" src="https://github.com/user-attachments/assets/1ffe8865-17e4-4fa4-95fc-7266283ddcaf" />





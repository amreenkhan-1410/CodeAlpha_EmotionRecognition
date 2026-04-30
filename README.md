# 🎙️ Emotion Recognition from Speech using Deep Learning

A final year Artificial Intelligence project that detects human emotions from speech audio using speech signal processing and deep learning techniques.

---

## 📌 Project Overview

**Emotion Recognition from Speech using Deep Learning** is a Speech Emotion Recognition (SER) system designed to analyze human voice recordings and predict the emotional state of the speaker.

The system extracts meaningful audio features from speech signals and uses a trained deep learning model to classify emotions such as:

- Happy
- Sad
- Angry
- Calm
- Neutral
- Fearful
- Disgust
- Surprised

The project also includes a **Streamlit-based real-time web application** where users can record speech through a microphone and instantly receive emotion predictions with confidence scores.

---

## 🎯 Objective

The main objective of this project is to build an intelligent speech-based emotion recognition system that can:

- Process raw speech audio files
- Extract important acoustic and spectral features
- Train a deep learning model on labeled emotional speech data
- Predict emotions from unseen speech samples
- Provide a simple and interactive web interface for real-time testing

This project demonstrates the practical use of **Machine Learning, Deep Learning, Digital Signal Processing, and Human-Computer Interaction**.

---

## 🛠️ Technologies Used

- **Python**
- **Librosa** - audio feature extraction and signal processing
- **NumPy** - numerical computation
- **Pandas** - dataset handling and preprocessing
- **TensorFlow / Keras** - deep learning model development
- **Scikit-learn** - data splitting, encoding, and evaluation utilities
- **Streamlit** - interactive web application
- **audio-recorder-streamlit** - microphone-based speech recording
- **SoundFile** - audio file handling
- **Joblib** - saving and loading preprocessing objects

---

## 📂 Dataset Information

The project uses the **RAVDESS Emotional Speech Audio Dataset**.

**RAVDESS** stands for **Ryerson Audio-Visual Database of Emotional Speech and Song**. It is a widely used benchmark dataset for speech emotion recognition research.

### Dataset Details

- Contains emotional speech recordings from **24 professional actors**
- Includes both male and female speakers
- Audio files are provided in `.wav` format
- Each audio file is labeled with an emotion category
- Supports supervised learning for emotion classification

### Emotion Classes Used

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## 🎧 Speech Features Extracted

To represent speech signals effectively, the project extracts multiple audio features using **Librosa**.

| Feature | Description |
|---|---|
| **MFCC** | Captures the short-term power spectrum of speech and represents vocal characteristics |
| **Chroma** | Represents pitch class information and harmonic content |
| **Mel Spectrogram** | Converts audio into the Mel scale to model human hearing perception |
| **Spectral Contrast** | Measures the difference between peaks and valleys in the audio spectrum |
| **Tonnetz** | Captures tonal centroid features related to harmonic relationships |

These features are combined into a numerical feature vector and used as input to the deep learning model.

---

## 🧠 Deep Learning Model Architecture

The project uses a **Deep Dense Neural Network** built with TensorFlow/Keras.

### Model Highlights

- Fully connected deep neural network
- Dense layers for high-level feature learning
- **Batch Normalization** for faster and more stable training
- **Dropout** for reducing overfitting
- Softmax output layer for multi-class emotion classification
- Final achieved accuracy: **approximately 69%**

### General Architecture

```text
Input Audio Features
        ↓
Dense Layer
        ↓
Batch Normalization
        ↓
Dropout
        ↓
Dense Layer
        ↓
Batch Normalization
        ↓
Dropout
        ↓
Output Layer with Softmax Activation
        ↓
Predicted Emotion
```

---

## 📁 Project Folder Structure

```text
EmotionRecognition/
│
├── app.py                    # Streamlit real-time web application
├── feature_extraction.py     # Audio feature extraction logic
├── prepare_dataset.py        # Dataset preprocessing and CSV generation
├── train_model.py            # Deep learning model training script
├── predict_emotion.py        # Sample emotion prediction script
├── requirements.txt          # Required Python dependencies
│
├── dataset/                  # RAVDESS dataset files
├── models/                   # Saved trained model and preprocessing files
├── output/                   # Generated outputs such as processed data/results
│
└── README.md                 # Project documentation
```

---

## ⚙️ Installation Steps

Follow the steps below to set up the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/EmotionRecognition.git
cd EmotionRecognition
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### Step 1: Prepare the Dataset

Place the RAVDESS `.wav` files inside the `dataset/` directory, then run:

```bash
python prepare_dataset.py
```

This script extracts labels and prepares the dataset for training.

### Step 2: Train the Deep Learning Model

```bash
python train_model.py
```

The trained model and required preprocessing files will be saved inside the `models/` directory.

### Step 3: Test Emotion Prediction on a Sample Audio File

```bash
python predict_emotion.py
```

This script loads the trained model and predicts the emotion for a sample speech audio input.

### Step 4: Run the Streamlit Web Application

```bash
streamlit run app.py
```

After running the command, open the local Streamlit URL shown in the terminal to use the application.

---

## 📊 Sample Output / Application Features

The Streamlit application provides an interactive interface for real-time speech emotion recognition.

### Key Features

- Manual microphone speech input
- Real-time audio recording
- Live emotion prediction
- Confidence score for predicted emotion
- Emotion probability distribution
- Clean and user-friendly web interface

### Example Output

```text
Predicted Emotion: Happy
Confidence Score: 82.45%
```

### Emotion Probability Distribution

```text
Happy      : 82.45%
Calm       : 06.32%
Neutral    : 04.18%
Sad        : 02.90%
Angry      : 01.85%
Fearful    : 01.10%
Disgust    : 00.75%
Surprised  : 00.45%
```

---

## 🚀 Future Enhancements

- Improve model accuracy using CNN, LSTM, GRU, or hybrid deep learning architectures
- Train on additional datasets such as CREMA-D, TESS, and SAVEE
- Add noise reduction and silence removal for better real-world performance
- Support multilingual speech emotion recognition
- Add speaker-independent evaluation
- Deploy the application on Streamlit Cloud or Hugging Face Spaces
- Add visual analytics for waveform, spectrogram, and prediction history
- Implement real-time continuous speech monitoring

---

## 👨‍💻 Author

**Project:** Emotion Recognition from Speech using Deep Learning  
**Domain:** Machine Learning, Deep Learning, Speech Signal Processing  
**Dataset:** RAVDESS Emotional Speech Dataset  
**Model Accuracy:** Approximately 69%

Developed as a final year Artificial Intelligence project to demonstrate practical implementation of speech emotion recognition using deep learning.

---

## ⭐ Repository Highlights

If you found this project useful, consider starring the repository and using it as a reference for speech emotion recognition, audio classification, and applied deep learning projects.

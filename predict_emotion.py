import numpy as np
import joblib
from tensorflow.keras.models import load_model
from feature_extraction import extract_features

# Load trained model
model = load_model("models/emotion_model.h5")
encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

# Give path of test wav file
file_path = "dataset/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav"

# Extract features
features = extract_features(file_path)

# Scale features
features = scaler.transform([features])

# Predict
prediction = model.predict(features)

predicted_emotion = encoder.inverse_transform([np.argmax(prediction)])

print("===================================")
print("Predicted Emotion:", predicted_emotion[0])
print("===================================")
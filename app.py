import streamlit as st
import numpy as np
import joblib
import tempfile
import os
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
from feature_extraction import extract_features
from audio_recorder_streamlit import audio_recorder

# Load model files
model = load_model("models/emotion_model.h5")
encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤")

st.title("🎤 Real-Time Speech Emotion Recognition")
st.write("Record your voice for 4 to 6 seconds with clear emotional tone.")

audio_bytes = audio_recorder(
    text="🎙 Click and Record",
    recording_color="#ff4b4b",
    neutral_color="#6c757d",
    icon_name="microphone",
    icon_size="3x",
)

if audio_bytes:

    st.audio(audio_bytes, format="audio/wav")

    # Save temp recording
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        temp_audio_path = f.name

    # Reload and amplify audio
    audio, sr = librosa.load(temp_audio_path, sr=22050)

    # remove silence
    audio, _ = librosa.effects.trim(audio)

    # if too short warning
    duration = librosa.get_duration(y=audio, sr=sr)

    if duration < 2:
        st.warning("Please record at least 3 to 5 seconds of speech.")
    else:
        # normalize and amplify
        audio = librosa.util.normalize(audio) * 1.5

        sf.write(temp_audio_path, audio, sr)

        # Feature extraction
        features = extract_features(temp_audio_path)
        features = scaler.transform([features])

        # Prediction
        prediction = model.predict(features)[0]

        top_index = np.argmax(prediction)
        predicted_emotion = encoder.inverse_transform([top_index])[0]
        confidence = prediction[top_index] * 100

        st.success(f"Predicted Emotion: {predicted_emotion}")
        st.write(f"Confidence Score: {confidence:.2f}%")

        st.subheader("Emotion Probability Distribution")

        emotion_names = encoder.classes_

        for emo, prob in zip(emotion_names, prediction):
            st.progress(float(prob))
            st.write(f"{emo} : {prob*100:.2f}%")

    os.remove(temp_audio_path)
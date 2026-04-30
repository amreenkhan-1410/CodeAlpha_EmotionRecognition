import librosa
import numpy as np

def extract_features(file_path):

    audio, sample_rate = librosa.load(file_path, sr=22050)

    # Remove silence
    audio, _ = librosa.effects.trim(audio)

    # Normalize volume
    audio = librosa.util.normalize(audio)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate).T, axis=0)

    combined = np.hstack([mfcc, chroma, mel, contrast, tonnetz])

    return combined
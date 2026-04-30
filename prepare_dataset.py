import os
import pandas as pd
from feature_extraction import extract_features

dataset_path = "dataset/RAVDESS"

features = []
labels = []

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

print("Reading audio files... Please wait.")

for actor_folder in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor_folder)

    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):

                parts = file.split("-")
                emotion_code = parts[2]
                emotion_label = emotion_dict[emotion_code]

                file_path = os.path.join(actor_path, file)

                mfcc_feature = extract_features(file_path)

                features.append(mfcc_feature)
                labels.append(emotion_label)

                print("Processed:", file)

df = pd.DataFrame(features)
df["label"] = labels

df.to_csv("output/emotion_features.csv", index=False)

print("===================================")
print("Dataset preparation completed.")
print("emotion_features.csv saved in output folder.")
print("===================================")
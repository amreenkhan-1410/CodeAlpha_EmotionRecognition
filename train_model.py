import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Load dataset
df = pd.read_csv("output/emotion_features.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
joblib.dump(encoder, "models/label_encoder.pkl")

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")

y_categorical = to_categorical(y_encoded)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Build Dense Neural Network
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(y_categorical.shape[1], activation='softmax'))

# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=80,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

print("===================================")
print("Final Improved Accuracy:", accuracy * 100, "%")
print("===================================")

# Save
model.save("models/emotion_model.h5")
print("Model saved successfully.")
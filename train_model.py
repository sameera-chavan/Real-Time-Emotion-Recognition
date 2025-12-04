import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load new dataset
print("Loading dataset...")
df = pd.read_csv("emotion_dataset.csv")

# Convert emotion labels to numerical categories
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
df['emotion'] = df['emotion'].map(lambda x: emotions.index(x))

# Split features and labels
X = df.iloc[:, 1:].values  # Pixel values
y = df['emotion'].values   # Labels

# Normalize pixel values
X = X / 255.0  # Scale pixels between 0 and 1
X = X.reshape(-1, 48, 48, 1)  # Reshape for CNN input

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=len(emotions))

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
print("Building model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(emotions), activation='softmax')  # Output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training model...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64)

# Save the trained model
model.save("emotion_model.h5")
print("Model trained and saved as 'emotion_model.h5'")

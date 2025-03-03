# train_sticker_model.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

def prepare_dataset(data_dir):
    """Prepare dataset from images with labeled sticker positions."""
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(data_dir, filename))
            img = cv2.resize(img, (224, 224))  # Resize for model input
            images.append(img)
            # Assume label file (e.g., 'image1.txt') with 5 (x, y) coords per line
            with open(os.path.join(data_dir, filename.replace('.jpg', '.txt'))) as f:
                coords = [float(x) for x in f.read().split()]
                labels.append(coords)  # [x1, y1, x2, y2, ..., x5, y5]
    return np.array(images), np.array(labels)

def build_model():
    """Build a simple CNN for sticker detection."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)  # 5 fingertips * (x, y) = 10 outputs
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    """Train the sticker detection model."""
    data_dir = 'sticker_dataset'  # Directory with images and labels
    images, labels = prepare_dataset(data_dir)
    images = images / 255.0  # Normalize

    model = build_model()
    model.fit(images, labels, epochs=10, batch_size=32, validation_split=0.2)
    model.save('sticker_detector.h5')
    print("Model saved as 'sticker_detector.h5'")

if __name__ == "__main__":
    main()

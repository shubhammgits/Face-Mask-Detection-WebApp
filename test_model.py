import os
import tensorflow as tf
import cv2

print("Testing model loading...")

# Check if files exist
model_path = 'best_mask_model.h5'
cascade_path = 'haarcascade_frontalface_default.xml'

print(f"Model file exists: {os.path.exists(model_path)}")
print(f"Cascade file exists: {os.path.exists(cascade_path)}")

# Try to load the model
try:
    if os.path.exists(model_path):
        print("Loading model...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    else:
        print("Model file not found")
except Exception as e:
    print(f"Error loading model: {e}")

# Try to load the cascade
try:
    if os.path.exists(cascade_path):
        print("Loading cascade...")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        print("Cascade loaded successfully!")
    else:
        print("Cascade file not found")
except Exception as e:
    print(f"Error loading cascade: {e}")
import os

# Check if files exist
print("Checking if required files exist...")
model_path = 'best_mask_model.h5'
cascade_path = 'haarcascade_frontalface_default.xml'

print(f"Model file exists: {os.path.exists(model_path)}")
print(f"Cascade file exists: {os.path.exists(cascade_path)}")

if os.path.exists(model_path):
    print(f"Model file size: {os.path.getsize(model_path)} bytes")

if os.path.exists(cascade_path):
    print(f"Cascade file size: {os.path.getsize(cascade_path)} bytes")

# Try to import required modules
print("\nChecking if required modules can be imported...")

try:
    import tensorflow as tf
    print("✓ TensorFlow imported successfully")
    print(f"  TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")

try:
    import cv2
    print("✓ OpenCV imported successfully")
    print(f"  OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
    print(f"  NumPy version: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

print("\nTest completed.")
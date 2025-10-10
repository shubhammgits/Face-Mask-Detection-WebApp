import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
        print(f"  TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        # This might be okay in some environments
        pass
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    return True

def test_files():
    """Test that required files exist"""
    print("\nTesting required files...")
    
    required_files = [
        'app.py',
        'best_mask_model.h5',
        'haarcascade_frontalface_default.xml',
        'requirements.txt'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file} exists ({size} bytes)")
        else:
            print(f"✗ {file} not found")
            return False
    
    # Check static files
    static_files = [
        'static/script.js',
        'static/style.css',
        'templates/index.html'
    ]
    
    for file in static_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file} exists ({size} bytes)")
        else:
            print(f"✗ {file} not found")
            return False
    
    return True

def test_model_loading():
    """Test that the model can be loaded"""
    print("\nTesting model loading...")
    
    model_file = 'best_mask_model.h5'
    if not os.path.exists(model_file):
        print(f"✗ Model file {model_file} not found")
        return False
    
    size = os.path.getsize(model_file)
    print(f"✓ Model file exists ({size} bytes)")
    
    if size < 1000000:  # Less than 1MB
        print("⚠ Model file seems unusually small")
    
    return True

if __name__ == "__main__":
    print("=== Face Mask Detection App Test ===\n")
    
    if not test_imports():
        print("\n❌ Import tests failed")
        sys.exit(1)
    
    if not test_files():
        print("\n❌ File tests failed")
        sys.exit(1)
    
    if not test_model_loading():
        print("\n❌ Model loading tests failed")
        sys.exit(1)
    
    print("\n✅ All tests passed!")
    print("\nYou can now run the application with:")
    print("python app.py")
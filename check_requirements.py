#!/usr/bin/env python3

import sys
import importlib

def check_package(package_name, version=None):
    try:
        module = importlib.import_module(package_name)
        if version and hasattr(module, '__version__'):
            print(f"✓ {package_name} ({module.__version__})")
        else:
            print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} (NOT FOUND)")
        return False

def main():
    print("Checking required packages...")
    print("=" * 40)
    
    required_packages = [
        'flask',
        'flask_cors',
        'tensorflow',
        'cv2',
        'numpy',
        'PIL',
        'werkzeug'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if not check_package(package):
            missing_packages.append(package)
    
    print("=" * 40)
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return 1
    else:
        print("All required packages are available!")
        return 0

if __name__ == '__main__':
    sys.exit(main())
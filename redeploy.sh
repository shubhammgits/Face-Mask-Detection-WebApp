#!/bin/bash

# Redeploy script for Face Mask Detection Web App

echo "Starting redeployment process..."

# Check if we're in the correct directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Please run this script from the project root directory."
    exit 1
fi

echo "Creating new deployment..."

# Commit changes to git (if using git)
if [ -d ".git" ]; then
    echo "Committing changes..."
    git add .
    git commit -m "Fix frontend issues and improve live detection"
    echo "Changes committed"
fi

echo "Redeployment script completed!"
echo "Please push to your Render repository to trigger deployment:"
echo "git push origin main"
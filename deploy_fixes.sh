#!/bin/bash

# Deployment script for face mask detection fixes

echo "Deploying face mask detection fixes..."

# Check if we're in the correct directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Please run this script from the project root directory."
    exit 1
fi

# Add all changes to git
echo "Adding changes to git..."
git add .

# Commit the changes
echo "Committing changes..."
git commit -m "Fix detection box rendering issues - ensure red/green frames display correctly on live server"

# Push to origin main
echo "Pushing to repository..."
git push origin main

echo "Deployment initiated! Check your Render dashboard for deployment status."
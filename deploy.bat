@echo off
title Face Mask Detection App Deployment

echo === Face Mask Detection App Deployment ===
echo This script will help you redeploy your application to Render

REM Check if we're in the right directory
if not exist "app.py" (
    echo Error: app.py not found. Please run this script from the project root directory.
    pause
    exit /b 1
)

echo Current directory: %cd%
echo Found required files:
echo   - app.py
echo   - Dockerfile
echo   - render.yaml
echo   - requirements.txt
echo   - best_mask_model.h5: 
if exist "best_mask_model.h5" (echo YES) else (echo NO)
echo   - haarcascade_frontalface_default.xml: 
if exist "haarcascade_frontalface_default.xml" (echo YES) else (echo NO)

echo.
echo === Deployment Steps ===
echo 1. Commit your changes to Git
echo 2. Push to your GitHub repository
echo 3. Go to your Render dashboard
echo 4. Find your web service
echo 5. Click 'Manual Deploy' -^> 'Deploy latest commit'
echo.
echo === Additional Notes ===
echo - Make sure your GitHub repository is connected to Render
echo - The application will automatically build using the Dockerfile
echo - Check the logs in Render dashboard for any deployment issues
echo - The app should be available at your Render URL after deployment

echo.
echo === Troubleshooting ===
echo If you still have issues:
echo 1. Check Render logs for detailed error messages
echo 2. Verify that all required files are in your GitHub repository
echo 3. Make sure the model file (best_mask_model.h5) is included
echo 4. Check that render.yaml configuration is correct

echo.
echo Deployment script completed.
pause
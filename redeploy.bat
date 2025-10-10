@echo off
REM Redeploy script for Face Mask Detection Web App

echo Starting redeployment process...

REM Check if we're in the correct directory
if not exist "app.py" (
    echo Error: app.py not found. Please run this script from the project root directory.
    exit /b 1
)

echo Creating new deployment...

REM Commit changes to git (if using git)
if exist ".git" (
    echo Committing changes...
    git add .
    git commit -m "Fix frontend issues and improve live detection"
    echo Changes committed
)

echo Redeployment script completed!
echo Please push to your Render repository to trigger deployment:
echo git push origin main

pause
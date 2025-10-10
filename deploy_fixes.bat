@echo off
REM Deployment script for face mask detection fixes

echo Deploying face mask detection fixes...

REM Check if we're in the correct directory
if not exist "app.py" (
    echo Error: app.py not found. Please run this script from the project root directory.
    pause
    exit /b 1
)

REM Add all changes to git
echo Adding changes to git...
git add .

REM Commit the changes
echo Committing changes...
git commit -m "Fix detection box rendering issues - ensure red/green frames display correctly on live server"

REM Push to origin main
echo Pushing to repository...
git push origin main

echo Deployment initiated! Check your Render dashboard for deployment status.

pause
@echo off
echo ====================================
echo Building SporesCounterToolByLO
echo ====================================
echo.

echo Step 1: Installing dependencies...
pip install -r requirements.txt
echo.

echo Step 2: Building executable...
pyinstaller SporesCounterToolByLO.spec --clean
echo.

if exist "dist\SporesCounterToolByLO.exe" (
    echo ====================================
    echo Build successful!
    echo ====================================
    echo.
    echo Your executable is located at:
    echo dist\SporesCounterToolByLO.exe
    echo.
    echo You can now:
    echo 1. Copy SporesCounterToolByLO.exe to any Windows computer
    echo 2. Double-click to run (no Python installation needed!)
    echo.
) else (
    echo ====================================
    echo Please check any error messages above.
)

pause

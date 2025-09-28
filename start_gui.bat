@echo off
echo ===============================================
echo VibeVoice GUI - Clean Startup
echo ===============================================

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Run the clean startup script
echo Starting VibeVoice GUI with port cleanup...
python start_gui.py %*

echo.
echo GUI has stopped. Press any key to exit...
pause >nul

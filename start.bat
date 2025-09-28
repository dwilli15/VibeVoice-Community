@echo off
REM VibeVoice Windows Launcher
REM This script provides easy access to all VibeVoice interfaces

title VibeVoice Launcher

echo.
echo  ██╗   ██╗██╗██████╗ ███████╗██╗   ██╗ ██████╗ ██╗ ██████╗███████╗
echo  ██║   ██║██║██╔══██╗██╔════╝██║   ██║██╔═══██╗██║██╔════╝██╔════╝
echo  ██║   ██║██║██████╔╝█████╗  ██║   ██║██║   ██║██║██║     █████╗  
echo  ╚██╗ ██╔╝██║██╔══██╗██╔══╝  ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝  
echo   ╚████╔╝ ██║██████╔╝███████╗ ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗
echo    ╚═══╝  ╚═╝╚═════╝ ╚══════╝  ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝
echo.
echo                      Voice Synthesis Platform
echo.

REM Try to activate local virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

:menu
echo ==========================================
echo            Choose an option:
echo ==========================================
echo.
echo  1. Setup Assistant (First time setup)
echo  2. Desktop GUI (Native interface)
echo  3. Web Interface (Browser-based)
echo  4. Container Manager (Docker operations)
echo  5. Open Output Folder
echo  6. Check System Status
echo  7. Help & Documentation
echo  8. Exit
echo.
set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto desktop
if "%choice%"=="3" goto web
if "%choice%"=="4" goto container
if "%choice%"=="5" goto output
if "%choice%"=="6" goto status
if "%choice%"=="7" goto help
if "%choice%"=="8" goto exit
echo Invalid choice. Please try again.
goto menu

:setup
echo.
echo Starting Setup Assistant...
python setup_gui.py
if errorlevel 1 (
    echo Failed to start setup assistant. Trying alternative method...
    py setup_gui.py
)
goto menu

:desktop
echo.
echo Starting Desktop GUI...
python desktop_gui.py
if errorlevel 1 (
    echo Failed to start desktop GUI. Trying alternative method...
    py desktop_gui.py
)
goto menu

:web
echo.
echo Checking web interface...
python launcher.py
if errorlevel 1 (
    echo Failed to start launcher. Trying alternative method...
    py launcher.py
)
goto menu

:container
echo.
echo Container Management Menu:
echo 1. Start Container
echo 2. Stop Container  
echo 3. Restart Container
echo 4. View Logs
echo 5. Back to main menu
echo.
set /p containerChoice="Enter choice (1-5): "

if "%containerChoice%"=="1" (
    echo Starting container...
    powershell -ExecutionPolicy Bypass -File manage-container.ps1 start
    pause
)
if "%containerChoice%"=="2" (
    echo Stopping container...
    powershell -ExecutionPolicy Bypass -File manage-container.ps1 stop
    pause
)
if "%containerChoice%"=="3" (
    echo Restarting container...
    powershell -ExecutionPolicy Bypass -File manage-container.ps1 restart
    pause
)
if "%containerChoice%"=="4" (
    echo Viewing logs...
    docker logs -f vibe
    pause
)
if "%containerChoice%"=="5" goto menu

goto menu

:output
echo.
echo Opening output folder...
if exist "%USERPROFILE%\VibeVoice_Output" (
    start "" "%USERPROFILE%\VibeVoice_Output"
) else (
    mkdir "%USERPROFILE%\VibeVoice_Output"
    start "" "%USERPROFILE%\VibeVoice_Output"
    echo Created output folder at %USERPROFILE%\VibeVoice_Output
)
pause
goto menu

:status
echo.
echo ==========================================
echo           System Status Check
echo ==========================================
echo.

REM Check Python
python --version 2>nul
if errorlevel 1 (
    echo ❌ Python not found
    py --version 2>nul
    if errorlevel 1 (
        echo ❌ Python not available
    ) else (
        echo ✅ Python available (py launcher)
    )
) else (
    echo ✅ Python available
)

REM Check Docker
docker --version 2>nul
if errorlevel 1 (
    echo ❌ Docker not found
) else (
    echo ✅ Docker available
)

REM Check GPU
nvidia-smi 2>nul
if errorlevel 1 (
    echo ⚠️  NVIDIA GPU not detected
) else (
    echo ✅ NVIDIA GPU available
)

REM Check container status
docker ps -a --filter "name=vibe" --format "table {{.Names}}\t{{.Status}}" 2>nul | findstr "vibe" >nul
if errorlevel 1 (
    echo ⚪ Container not found
) else (
    echo 📊 Container status:
    docker ps -a --filter "name=vibe" --format "table {{.Names}}\t{{.Status}}"
)

REM Check port 7860
netstat -an | findstr ":7860" >nul
if errorlevel 1 (
    echo 🔴 Port 7860 not in use
) else (
    echo 🟢 Port 7860 is active
)

echo.
pause
goto menu

:help
echo.
echo ==========================================
echo          VibeVoice Help & Tips
echo ==========================================
echo.
echo Quick Start:
echo 1. Run "Setup Assistant" for first-time installation
echo 2. Use "Desktop GUI" for local processing
echo 3. Use "Web Interface" for browser-based access
echo.
echo Troubleshooting:
echo - If Python scripts fail, try installing dependencies:
echo   pip install -r requirements-gui.txt
echo - For GPU issues, ensure NVIDIA drivers are installed
echo - For Docker issues, install Docker Desktop
echo.
echo File Locations:
echo - Output: %USERPROFILE%\VibeVoice_Output
echo - Logs: Use Container Manager to view
echo.
echo Documentation:
echo - README.md (Project overview)
echo - CONTAINER_GUIDE.md (Docker setup)
echo.
pause
goto menu

:exit
echo.
echo Thank you for using VibeVoice!
echo.
pause
exit /b 0

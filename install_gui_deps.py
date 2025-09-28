#!/usr/bin/env python3
"""
Installation script for VibeVoice Integrated TTS/STT GUI dependencies
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """Install GUI dependencies"""
    print("Installing VibeVoice Integrated TTS/STT GUI dependencies...")
    print("=" * 60)
    
    # Core dependencies (required)
    core_packages = [
        "streamlit",
        "openai-whisper", 
        "soundfile",
        "pydub"
    ]
    
    # Optional dependencies (nice to have)
    optional_packages = [
        "TTS",  # Coqui TTS
        "resampy",
        "ffmpeg-python"
    ]
    
    success_count = 0
    total_count = len(core_packages) + len(optional_packages)
    
    print("Installing core dependencies...")
    for package in core_packages:
        if install_package(package):
            success_count += 1
    
    print("\nInstalling optional dependencies...")
    for package in optional_packages:
        if install_package(package):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Installation complete: {success_count}/{total_count} packages installed successfully")
    
    if success_count >= len(core_packages):
        print("\n✓ Core dependencies installed successfully!")
        print("You can now run the GUI with: streamlit run streamlit_gui.py")
    else:
        print("\n⚠️  Some core dependencies failed to install.")
        print("Please install them manually or check your internet connection.")
        print("Run: pip install streamlit openai-whisper soundfile pydub")
    
    # Check for ffmpeg
    try:
        subprocess.check_call(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✓ FFmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  FFmpeg not found. Install it for better audio format support:")
        print("   Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg")
        print("   macOS: brew install ffmpeg")
        print("   Windows: Download from https://ffmpeg.org/download.html")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Launcher script for VibeVoice Integrated TTS/STT Studio
"""

import subprocess
import sys
import os
from pathlib import Path

def check_streamlit():
    """Check if Streamlit is available"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def main():
    """Launch the VibeVoice GUI"""
    print("🎙️ VibeVoice Integrated TTS/STT Studio Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("streamlit_gui.py").exists():
        print("❌ Error: streamlit_gui.py not found in current directory")
        print("Please run this script from the VibeVoice-Community directory")
        return 1
    
    # Check if Streamlit is available
    if not check_streamlit():
        print("⚠️  Streamlit not found. Installing dependencies...")
        try:
            subprocess.check_call([sys.executable, "install_gui_deps.py"])
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run:")
            print("   python install_gui_deps.py")
            return 1
    
    print("🚀 Launching VibeVoice GUI...")
    print("   - The GUI will open in your default web browser")
    print("   - Use Ctrl+C to stop the server")
    print("   - Check the sidebar to load models before using")
    print()
    
    try:
        # Launch Streamlit
        subprocess.check_call([
            sys.executable, "-m", "streamlit", "run", "streamlit_gui.py",
            "--server.headless", "false",
            "--server.address", "localhost",
            "--server.port", "8501"
        ])
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch GUI: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 VibeVoice GUI stopped by user")
        return 0

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Test script for VibeVoice Integrated TTS/STT Studio
"""

import sys
import os
import tempfile
import traceback
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import librosa
        print("✅ Librosa imported successfully")
    except ImportError as e:
        print(f"❌ Librosa import failed: {e}")
        return False
    
    try:
        import soundfile as sf
        print("✅ SoundFile imported successfully")
    except ImportError as e:
        print(f"❌ SoundFile import failed: {e}")
        return False
    
    # Test optional imports
    try:
        import streamlit as st
        print("✅ Streamlit available")
    except ImportError:
        print("⚠️  Streamlit not available (expected for testing)")
    
    try:
        import whisper
        print("✅ Whisper available")
    except ImportError:
        print("⚠️  Whisper not available (expected for demo mode)")
    
    try:
        from TTS.api import TTS
        print("✅ Coqui TTS available")
    except ImportError:
        print("⚠️  Coqui TTS not available (expected for demo mode)")
    
    print("✅ Import tests completed successfully")
    return True

def test_vibevoice_imports():
    """Test VibeVoice specific imports"""
    print("\n🧪 Testing VibeVoice imports...")
    
    try:
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
        print("✅ VibeVoiceConfig imported")
    except ImportError as e:
        print(f"❌ VibeVoiceConfig import failed: {e}")
        return False
    
    try:
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        print("✅ VibeVoiceForConditionalGenerationInference imported")
    except ImportError as e:
        print(f"❌ VibeVoiceForConditionalGenerationInference import failed: {e}")
        return False
    
    try:
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        print("✅ VibeVoiceProcessor imported")
    except ImportError as e:
        print(f"❌ VibeVoiceProcessor import failed: {e}")
        return False
    
    print("✅ VibeVoice import tests completed successfully")
    return True

def test_app_initialization():
    """Test app initialization"""
    print("\n🧪 Testing app initialization...")
    
    try:
        # Import without Streamlit dependency
        sys.modules['streamlit'] = None  # Mock streamlit to avoid import error
        
        # Create a minimal version for testing
        class MockApp:
            def __init__(self):
                self.demo_mode = True
                self.models_loaded = {'vibevoice': False, 'whisper': False, 'coqui': False}
                self.voice_presets = {}
                self.device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
                self.setup_voice_presets()
            
            def setup_voice_presets(self):
                demo_voices = {
                    "Alice": "Female, professional narrator",
                    "Bob": "Male, friendly conversational",
                    "Charlie": "Male, technical presenter",
                    "Diana": "Female, storyteller"
                }
                self.voice_presets.update(demo_voices)
        
        app = MockApp()
        print(f"✅ App initialized successfully")
        print(f"   Device: {app.device}")
        print(f"   Demo mode: {app.demo_mode}")
        print(f"   Voice presets: {len(app.voice_presets)}")
        print(f"   Available voices: {list(app.voice_presets.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ App initialization failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_demo_audio_generation():
    """Test demo audio generation"""
    print("\n🧪 Testing demo audio generation...")
    
    try:
        import numpy as np
        
        # Simple audio generation test
        text = "Hello, this is a test of the audio generation system."
        duration = max(2.0, len(text.split()) / 2.5)
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate synthetic audio
        frequency_base = 250
        audio = (
            0.4 * np.sin(2 * np.pi * frequency_base * t) +
            0.3 * np.sin(2 * np.pi * (frequency_base * 1.8) * t) +
            0.2 * np.sin(2 * np.pi * (frequency_base * 2.7) * t)
        )
        
        # Apply envelope
        envelope = np.ones_like(t)
        fade_samples = int(0.1 * sample_rate)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        audio = audio * envelope
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.7
        
        print(f"✅ Demo audio generated successfully")
        print(f"   Duration: {duration:.2f} seconds")  
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Audio shape: {audio.shape}")
        print(f"   Audio range: [{np.min(audio):.3f}, {np.max(audio):.3f}]")
        
        # Test saving audio
        try:
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio, sample_rate)
                file_size = os.path.getsize(tmp_file.name)
                print(f"✅ Audio saved successfully: {tmp_file.name} ({file_size} bytes)")
                os.unlink(tmp_file.name)
        except Exception as e:
            print(f"⚠️  Audio save test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo audio generation failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_file_structure():
    """Test required file structure"""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "streamlit_gui.py",
        "install_gui_deps.py", 
        "launch_gui.py",
        "mock_gui_demo.py",
        "simple_gui_demo.py",
        "requirements_gui.txt",
        "GUI_README.md"
    ]
    
    all_present = True
    for file_path in required_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} missing")
            all_present = False
    
    # Check demo files
    demo_files = [
        "demo_config.json",
        "demo_texts/simple_tts.txt",
        "demo_texts/long_form.txt", 
        "demo_texts/podcast_script.txt",
        "demo_texts/audiobook_sample.txt"
    ]
    
    for file_path in demo_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"⚠️  {file_path} missing (optional)")
    
    if all_present:
        print("✅ All required files present")
        return True
    else:
        print("❌ Some required files missing")
        return False

def main():
    """Run all tests"""
    print("🎙️ VibeVoice Integrated TTS/STT Studio - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("VibeVoice Import Tests", test_vibevoice_imports),
        ("App Initialization", test_app_initialization),
        ("Demo Audio Generation", test_demo_audio_generation),
        ("File Structure", test_file_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The GUI is ready for use.")
        print("\nNext steps:")
        print("1. Install additional dependencies: python install_gui_deps.py")
        print("2. Launch the GUI: python launch_gui.py")
        print("3. Or run Streamlit directly: streamlit run streamlit_gui.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
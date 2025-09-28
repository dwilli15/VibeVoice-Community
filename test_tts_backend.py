#!/usr/bin/env python3
"""
Test script for TTS Backend functionality
Tests multi-engine detection, voice listing, and backend operations
"""

import sys
import traceback
from pathlib import Path

def test_tts_backend_import():
    """Test TTS backend imports"""
    print("🔍 Testing TTS backend imports...")
    
    try:
        from tts_backend import get_tts_backend, TTSRequest, Voice, TTSEngine
        print("✅ TTS backend classes imported successfully")
        return True
    except ImportError as e:
        print(f"❌ TTS backend import failed: {e}")
        traceback.print_exc()
        return False

def test_engine_detection():
    """Test engine availability detection"""
    print("\n🔍 Testing engine detection...")
    
    try:
        from tts_backend import VIBEVOICE_AVAILABLE, COQUI_AVAILABLE, TORCH_AVAILABLE
        
        print(f"📋 VibeVoice Available: {VIBEVOICE_AVAILABLE}")
        print(f"📋 Coqui Available: {COQUI_AVAILABLE}")
        print(f"📋 PyTorch Available: {TORCH_AVAILABLE}")
        
        # Test engine enumeration
        from tts_backend import TTSEngine
        engines = list(TTSEngine)
        print(f"📋 Available engine types: {[e.value for e in engines]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Engine detection failed: {e}")
        traceback.print_exc()
        return False

def test_backend_creation():
    """Test TTS backend creation"""
    print("\n🔍 Testing TTS backend creation...")
    
    try:
        from tts_backend import get_tts_backend
        
        # Try to create backend
        backend = get_tts_backend()
        print("✅ TTS backend created successfully")
        
        if backend:
            print(f"📋 Backend type: {type(backend).__name__}")
            
            # Test if backend has expected methods
            methods = ['list_voices', 'generate_speech']
            for method in methods:
                if hasattr(backend, method):
                    print(f"✅ Backend has {method} method")
                else:
                    print(f"⚠️ Backend missing {method} method")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend creation failed: {e}")
        traceback.print_exc()
        return False

def test_voice_listing():
    """Test voice listing functionality"""
    print("\n🔍 Testing voice listing...")
    
    try:
        from tts_backend import get_tts_backend
        
        backend = get_tts_backend()
        if backend and hasattr(backend, 'list_voices'):
            voices = backend.list_voices()
            print(f"✅ Voice listing successful - found {len(voices)} voices")
            
            # Show first few voices
            for i, voice in enumerate(voices[:5]):
                print(f"  🎤 {voice.name} ({voice.engine.value}) - {voice.language}")
            
            if len(voices) > 5:
                print(f"  ... and {len(voices) - 5} more voices")
                
            return True
        else:
            print("⚠️ Voice listing not available - backend doesn't support it")
            return True
            
    except Exception as e:
        print(f"❌ Voice listing failed: {e}")
        traceback.print_exc()
        return False

def test_simple_tts_backend():
    """Test simple TTS backend as fallback"""
    print("\n🔍 Testing simple TTS backend...")
    
    try:
        from simple_tts_backend import SimpleTTSBackend
        
        backend = SimpleTTSBackend()
        print("✅ Simple TTS backend created successfully")
        
        # Test voice listing
        voices = backend.list_voices()
        print(f"📋 Simple backend has {len(voices)} voices")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple TTS backend test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all TTS backend tests"""
    print("🧪 TTS Backend Functionality Tests")
    print("=" * 50)
    
    tests = [
        ("TTS Backend Import", test_tts_backend_import),
        ("Engine Detection", test_engine_detection),
        ("Backend Creation", test_backend_creation),
        ("Voice Listing", test_voice_listing),
        ("Simple TTS Backend", test_simple_tts_backend),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print(f"\n{'=' * 50}")
    print("📊 Test Summary:")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All TTS backend tests passed!")
        return True
    else:
        print("⚠️ Some tests failed - see details above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test script for VibeVoice core functionality
Tests configuration, processor, and basic model operations
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all core VibeVoice imports"""
    print("🔍 Testing VibeVoice imports...")
    
    try:
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
        print("✅ VibeVoiceConfig imported")
    except ImportError as e:
        print(f"❌ VibeVoiceConfig import failed: {e}")
        return False
    
    try:
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        print("✅ VibeVoiceProcessor imported")
    except ImportError as e:
        print(f"❌ VibeVoiceProcessor import failed: {e}")
        return False
    
    try:
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        print("✅ VibeVoiceForConditionalGenerationInference imported")
    except ImportError as e:
        print(f"❌ VibeVoiceForConditionalGenerationInference import failed: {e}")
        return False
        
    return True

def test_processor_creation():
    """Test processor creation without model loading"""
    print("\n🔍 Testing processor creation...")
    
    try:
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        
        # Test basic processor creation
        processor = VibeVoiceProcessor()
        print("✅ Basic VibeVoiceProcessor created successfully")
        
        # Test processor attributes
        print(f"📋 Speech token compression ratio: {processor.speech_tok_compress_ratio}")
        print(f"📋 DB normalize enabled: {processor.db_normalize}")
        print(f"📋 System prompt: {processor.system_prompt[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Processor creation failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    
    try:
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
        
        # Test default config creation
        config = VibeVoiceConfig()
        print("✅ Default VibeVoiceConfig created successfully")
        
        # Print some config attributes if available
        if hasattr(config, 'model_type'):
            print(f"📋 Model type: {config.model_type}")
        if hasattr(config, 'vocab_size'):
            print(f"📋 Vocab size: {config.vocab_size}")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_available_models():
    """Test if any models are available locally"""
    print("\n🔍 Checking for available models...")
    
    model_dirs = [
        Path("./models"),
        Path("./models/hf"),
        Path("./models/hf/hub")
    ]
    
    found_models = []
    for model_dir in model_dirs:
        if model_dir.exists():
            print(f"📁 Found model directory: {model_dir}")
            for item in model_dir.iterdir():
                if item.is_dir():
                    found_models.append(item)
                    print(f"  📂 {item.name}")
    
    if found_models:
        print(f"✅ Found {len(found_models)} potential model directories")
        return True
    else:
        print("⚠️ No model directories found - this is normal for fresh installations")
        return True

def main():
    """Run all tests"""
    print("🧪 VibeVoice Core Functionality Tests")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Processor Creation", test_processor_creation),
        ("Configuration Tests", test_configuration),
        ("Model Availability", test_available_models),
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
        print("🎉 All core functionality tests passed!")
        return True
    else:
        print("⚠️ Some tests failed - see details above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

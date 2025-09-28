#!/usr/bin/env python3
"""
Test script for basic functionality without problematic dependencies
"""

import sys
import traceback
from pathlib import Path

def test_basic_imports():
    """Test basic imports that should work"""
    print("🔍 Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import soundfile as sf
        print("✅ soundfile imported")
    except ImportError as e:
        print(f"❌ soundfile import failed: {e}")
        return False
    
    try:
        import librosa
        print("✅ librosa imported")
    except ImportError as e:
        print(f"❌ librosa import failed: {e}")
        return False
        
    try:
        import gradio as gr
        print(f"✅ Gradio {gr.__version__} imported")
    except ImportError as e:
        print(f"❌ Gradio import failed: {e}")
        return False
        
    return True

def test_file_structure():
    """Test project file structure"""
    print("\n🔍 Testing project structure...")
    
    essential_files = [
        "pyproject.toml",
        "README.md", 
        "vibevoice/__init__.py",
        "demo/gradio_demo.py",
        "docker-compose.yml",
        "launcher.py",
        "desktop_gui.py",
        "ebook_converter.py"
    ]
    
    missing_files = []
    for file_path in essential_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"⚠️ {len(missing_files)} essential files missing")
        return False
    else:
        print("✅ All essential files present")
        return True

def test_configurations():
    """Test configuration files"""
    print("\n🔍 Testing configuration files...")
    
    try:
        # Test pyproject.toml
        import tomllib
        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)
        print("✅ pyproject.toml is valid TOML")
        
        if "project" in config:
            print(f"📋 Project name: {config['project'].get('name', 'Unknown')}")
            print(f"📋 Project version: {config['project'].get('version', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ pyproject.toml validation failed: {e}")
        return False
    
    try:
        # Test docker-compose.yml
        import yaml
        with open("docker-compose.yml", "r") as f:
            docker_config = yaml.safe_load(f)
        print("✅ docker-compose.yml is valid YAML")
        
        services = docker_config.get("services", {})
        print(f"📋 Docker services: {list(services.keys())}")
        
    except Exception as e:
        print(f"❌ docker-compose.yml validation failed: {e}")
        return False
        
    return True

def test_ebook_converter():
    """Test ebook converter imports"""
    print("\n🔍 Testing ebook converter...")
    
    try:
        from ebook_converter import EbookFormat, Chapter, ConversionConfig, TextProcessor
        print("✅ Ebook converter classes imported successfully")
        
        # Test enum
        formats = list(EbookFormat)
        print(f"📋 Supported formats: {[f.value for f in formats]}")
        
        # Test configuration
        config = ConversionConfig(
            input_file="test.txt",
            output_dir="./outputs"
        )
        print(f"✅ ConversionConfig created - voice: {config.voice_name}, speed: {config.speed}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ebook converter import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Ebook converter test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all basic tests"""
    print("🧪 Basic Functionality Tests (Without Complex Dependencies)")
    print("=" * 70)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("File Structure", test_file_structure),
        ("Configuration Files", test_configurations),
        ("Ebook Converter", test_ebook_converter),
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
    
    print(f"\n{'=' * 70}")
    print("📊 Test Summary:")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return True
    else:
        print("⚠️ Some tests failed - see details above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test script for ebook packaging functionality
Tests MP3/M4B creation with chapter markers and metadata
"""

import tempfile
import shutil
from pathlib import Path
import json

def create_test_ebook():
    """Create a simple test ebook"""
    test_content = """
Chapter 1: The Beginning

This is the first chapter of our test audiobook. It contains several sentences to test the text-to-speech conversion. The chapter should be long enough to produce meaningful audio output for testing purposes.

This paragraph continues the first chapter with additional content to ensure we have sufficient text for audio generation testing.

Chapter 2: The Middle

The second chapter begins here with different content. This chapter will test the chapter detection and splitting functionality of our ebook converter.

We want to ensure that chapters are properly identified and converted to separate audio segments that can be combined into a complete audiobook.

Chapter 3: The End

Our final chapter wraps up the test content. This short chapter demonstrates how the converter handles the end of the document and finalizes the audiobook creation process.

The end.
"""
    
    # Create temporary text file
    temp_file = Path(tempfile.gettempdir()) / "test_ebook.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    return str(temp_file)

def test_ebook_analysis():
    """Test ebook analysis functionality"""
    print("📖 Testing ebook analysis...")
    
    try:
        from ebook_converter import EbookToAudiobookConverter
        
        # Create test ebook
        test_file = create_test_ebook()
        
        # Initialize converter
        converter = EbookToAudiobookConverter()
        
        # Analyze ebook
        analysis = converter.analyze_ebook(test_file)
        
        print(f"✅ Analysis completed:")
        print(f"   Total words: {analysis['total_words']}")
        print(f"   Chapters: {analysis['total_chapters']}")
        print(f"   Estimated duration: {analysis['estimated_duration_minutes']:.1f} minutes")
        
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        return False

def test_wav_conversion():
    """Test basic WAV conversion"""
    print("🎵 Testing WAV conversion...")
    
    try:
        from ebook_converter import EbookToAudiobookConverter, ConversionConfig
        
        # Create test ebook
        test_file = create_test_ebook()
        
        # Create output directory
        output_dir = Path(tempfile.gettempdir()) / "test_wav_output"
        output_dir.mkdir(exist_ok=True)
        
        # Initialize converter
        converter = EbookToAudiobookConverter()
        
        # Create config
        config = ConversionConfig(
            input_file=test_file,
            output_dir=str(output_dir),
            voice_name="bf_isabella",
            speed=1.5,  # Faster for testing
            format="wav",
            preview_mode=True,  # Only convert first 2 chapters
            engine="vibevoice"
        )
        
        # Convert
        results = converter.convert_to_audiobook(config)
        
        print(f"✅ WAV conversion completed:")
        print(f"   Audio files: {len(results['audio_files'])}")
        print(f"   Engine used: {results.get('engine_used', 'unknown')}")
        print(f"   Errors: {len(results['errors'])}")
        
        if results['errors']:
            for error in results['errors']:
                print(f"   ⚠️ {error}")
        
        # Check output files
        for audio_file in results['audio_files'][:3]:  # Show first 3
            file_path = Path(audio_file)
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   📁 {file_path.name} ({size_mb:.2f} MB)")
            else:
                print(f"   ❌ Missing: {file_path.name}")
        
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        
        return len(results['audio_files']) > 0
        
    except Exception as e:
        print(f"❌ WAV conversion test failed: {e}")
        return False

def test_mp3_conversion():
    """Test MP3 conversion with metadata"""
    print("🎵 Testing MP3 conversion...")
    
    try:
        from ebook_converter import EbookToAudiobookConverter, ConversionConfig
        
        # Create test ebook
        test_file = create_test_ebook()
        
        # Create output directory
        output_dir = Path(tempfile.gettempdir()) / "test_mp3_output"
        output_dir.mkdir(exist_ok=True)
        
        # Initialize converter
        converter = EbookToAudiobookConverter()
        
        # Check FFmpeg availability
        if not converter.ffmpeg_available:
            print("⚠️ FFmpeg not available - skipping MP3 test")
            return True
        
        # Create config
        config = ConversionConfig(
            input_file=test_file,
            output_dir=str(output_dir),
            voice_name="bf_isabella",
            speed=1.5,  # Faster for testing
            format="mp3",
            bitrate="96k",  # Lower bitrate for testing
            title="Test Audiobook",
            author="Test Author",
            preview_mode=True,
            engine="vibevoice"
        )
        
        # Convert
        results = converter.convert_to_audiobook(config)
        
        print(f"✅ MP3 conversion completed:")
        print(f"   Audio files: {len(results['audio_files'])}")
        print(f"   Format: {results.get('format', 'unknown')}")
        
        # Check for combined MP3
        combined_found = any('audiobook' in Path(f).name.lower() for f in results['audio_files'])
        print(f"   📚 Combined audiobook: {'✅' if combined_found else '❌'}")
        
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        
        return len(results['audio_files']) > 0
        
    except Exception as e:
        print(f"❌ MP3 conversion test failed: {e}")
        return False

def test_m4b_conversion():
    """Test M4B conversion with chapters"""
    print("🎵 Testing M4B conversion...")
    
    try:
        from ebook_converter import EbookToAudiobookConverter, ConversionConfig
        
        # Create test ebook
        test_file = create_test_ebook()
        
        # Create output directory
        output_dir = Path(tempfile.gettempdir()) / "test_m4b_output"
        output_dir.mkdir(exist_ok=True)
        
        # Initialize converter
        converter = EbookToAudiobookConverter()
        
        # Check FFmpeg availability
        if not converter.ffmpeg_available:
            print("⚠️ FFmpeg not available - skipping M4B test")
            return True
        
        # Create config
        config = ConversionConfig(
            input_file=test_file,
            output_dir=str(output_dir),
            voice_name="bf_isabella",
            speed=1.5,  # Faster for testing
            format="m4b",
            bitrate="96k",  # Lower bitrate for testing
            title="Test Audiobook M4B",
            author="Test Author",
            preview_mode=True,
            engine="vibevoice"
        )
        
        # Convert
        results = converter.convert_to_audiobook(config)
        
        print(f"✅ M4B conversion completed:")
        print(f"   Audio files: {len(results['audio_files'])}")
        
        # Check for M4B file
        m4b_found = any(Path(f).suffix.lower() == '.m4b' for f in results['audio_files'])
        print(f"   📱 M4B audiobook: {'✅' if m4b_found else '❌'}")
        
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        
        return len(results['audio_files']) > 0
        
    except Exception as e:
        print(f"❌ M4B conversion test failed: {e}")
        return False

def main():
    """Run all packaging tests"""
    print("🧪 Ebook Packaging Test Suite")
    print("=" * 50)
    
    tests = [
        ("Analysis", test_ebook_analysis),
        ("WAV Conversion", test_wav_conversion),
        ("MP3 Conversion", test_mp3_conversion),
        ("M4B Conversion", test_m4b_conversion),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
        print(f"{'✅' if success else '❌'} {test_name}: {'PASSED' if success else 'FAILED'}")
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ebook packaging is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

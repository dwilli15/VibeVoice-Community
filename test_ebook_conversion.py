#!/usr/bin/env python3
"""
Test script for ebook conversion functionality
Tests format support, text processing, and conversion options
"""

import sys
import tempfile
import os
from pathlib import Path

def test_ebook_imports():
    """Test ebook converter imports and dependencies"""
    print("🔍 Testing ebook converter dependencies...")
    
    # Test core imports
    try:
        from ebook_converter import EbookFormat, Chapter, ConversionConfig, TextProcessor
        print("✅ Core ebook converter classes imported")
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False
    
    # Test optional format dependencies
    format_support = {}
    
    try:
        import pypdf
        format_support['PDF'] = True
        print("✅ PDF support available (pypdf)")
    except ImportError:
        format_support['PDF'] = False
        print("⚠️ PDF support not available (pypdf missing)")
    
    try:
        from docx import Document
        format_support['DOCX'] = True
        print("✅ DOCX support available")
    except ImportError:
        format_support['DOCX'] = False
        print("⚠️ DOCX support not available (python-docx missing)")
    
    try:
        import ebooklib
        from bs4 import BeautifulSoup
        format_support['EPUB'] = True
        print("✅ EPUB support available")
    except ImportError:
        format_support['EPUB'] = False
        print("⚠️ EPUB support not available (ebooklib/beautifulsoup4 missing)")
    
    # TXT is always supported
    format_support['TXT'] = True
    print("✅ TXT support always available")
    
    print(f"📋 Format support summary: {sum(format_support.values())}/4 formats supported")
    return True

def test_text_processor():
    """Test text processing functionality"""
    print("\n🔍 Testing text processor...")
    
    try:
        from ebook_converter import TextProcessor
        
        processor = TextProcessor()
        print("✅ TextProcessor created successfully")
        
        # Test text cleaning
        test_text = "  This   is    a  test.\n\n\nWith   extra   spaces.  "
        cleaned = processor.clean_text(test_text)
        print(f"📋 Text cleaning: '{test_text[:30]}...' → '{cleaned[:30]}...'")
        
        # Test sentence splitting (basic test)
        if hasattr(processor, 'split_sentences'):
            sentences = processor.split_sentences("Hello world! How are you? Fine, thanks.")
            print(f"📋 Sentence splitting: {len(sentences)} sentences detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Text processor test failed: {e}")
        return False

def test_conversion_config():
    """Test conversion configuration"""
    print("\n🔍 Testing conversion configuration...")
    
    try:
        from ebook_converter import ConversionConfig, EbookFormat
        
        # Test default config
        config = ConversionConfig(
            input_file="test.txt",
            output_dir="./outputs"
        )
        
        print(f"✅ Default config created")
        print(f"📋 Voice: {config.voice_name}")
        print(f"📋 Speed: {config.speed}")
        print(f"📋 Format: {config.format}")
        print(f"📋 Engine: {config.engine}")
        
        # Test custom config
        custom_config = ConversionConfig(
            input_file="book.epub",
            output_dir="./custom_output",
            voice_name="custom_voice",
            speed=1.5,
            format="mp3",
            engine="coqui"
        )
        
        print(f"✅ Custom config created with voice: {custom_config.voice_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_chapter_structure():
    """Test chapter data structure"""
    print("\n🔍 Testing chapter structure...")
    
    try:
        from ebook_converter import Chapter
        
        # Test chapter creation
        chapter = Chapter(
            title="Test Chapter",
            content="This is a test chapter with some content.",
            chapter_number=1,
            word_count=9,
            estimated_duration=0.5
        )
        
        print(f"✅ Chapter created: '{chapter.title}'")
        print(f"📋 Word count: {chapter.word_count}")
        print(f"📋 Estimated duration: {chapter.estimated_duration} minutes")
        
        return True
        
    except Exception as e:
        print(f"❌ Chapter test failed: {e}")
        return False

def test_file_format_detection():
    """Test file format detection"""
    print("\n🔍 Testing file format detection...")
    
    try:
        from ebook_converter import EbookFormat
        
        # Test format enumeration
        formats = list(EbookFormat)
        print(f"✅ Available formats: {[f.value for f in formats]}")
        
        # Test format detection logic (if exists)
        test_files = {
            "book.pdf": "pdf",
            "document.docx": "docx", 
            "story.epub": "epub",
            "text.txt": "txt"
        }
        
        for filename, expected_format in test_files.items():
            extension = Path(filename).suffix.lower()[1:]  # Remove the dot
            if extension in [f.value for f in formats]:
                print(f"✅ {filename} → {extension} format detected")
            else:
                print(f"⚠️ {filename} → format not supported")
        
        return True
        
    except Exception as e:
        print(f"❌ Format detection test failed: {e}")
        return False

def test_sample_text_conversion():
    """Test sample text conversion (dry run)"""
    print("\n🔍 Testing sample text conversion...")
    
    try:
        from ebook_converter import ConversionConfig, TextProcessor
        
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""Chapter 1: The Beginning

This is the first chapter of our test book. 
It contains multiple sentences and paragraphs.

Chapter 2: The Middle

This is the second chapter with more content.
We can test how the text processing works.

Chapter 3: The End

And this is the final chapter of our sample book.
It concludes our testing story.""")
            temp_file = f.name
        
        print(f"✅ Created temporary test file: {temp_file}")
        
        # Test configuration for this file
        config = ConversionConfig(
            input_file=temp_file,
            output_dir=tempfile.gettempdir(),
            preview_mode=True,
            preview_chapters=2
        )
        
        print(f"✅ Configuration created for conversion")
        print(f"📋 Preview mode: {config.preview_mode}")
        print(f"📋 Preview chapters: {config.preview_chapters}")
        
        # Test text processor
        processor = TextProcessor()
        
        with open(temp_file, 'r') as f:
            content = f.read()
        
        cleaned_content = processor.clean_text(content)
        print(f"📋 Original length: {len(content)} chars")
        print(f"📋 Cleaned length: {len(cleaned_content)} chars")
        
        # Cleanup
        os.unlink(temp_file)
        print("✅ Sample conversion test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample conversion test failed: {e}")
        return False

def main():
    """Run all ebook conversion tests"""
    print("🧪 Ebook Conversion Pipeline Tests")
    print("=" * 50)
    
    tests = [
        ("Ebook Imports", test_ebook_imports),
        ("Text Processor", test_text_processor),
        ("Conversion Config", test_conversion_config),
        ("Chapter Structure", test_chapter_structure),
        ("Format Detection", test_file_format_detection),
        ("Sample Conversion", test_sample_text_conversion),
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
        print("🎉 All ebook conversion tests passed!")
        return True
    else:
        print("⚠️ Some tests failed - see details above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

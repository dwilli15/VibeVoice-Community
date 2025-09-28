"""Test script for the ebook converter"""

from ebook_converter import EbookToAudiobookConverter
from pathlib import Path

def test_converter():
    print('🧪 Testing Ebook Converter...')

    # Create test converter
    converter = EbookToAudiobookConverter()
    print('✅ Converter initialized')
    print(f'TTS Available: {converter.tts_available}')

    # Create a simple test text file
    test_file = Path('test_ebook.txt')
    test_content = """Chapter 1: Introduction

This is a test ebook for the VibeVoice audiobook converter.
It contains multiple sentences to test the text processing capabilities.
The converter should be able to handle this content properly.

Chapter 2: Features

The ebook converter supports multiple formats including PDF, TXT, DOCX, and EPUB.
It uses advanced text processing to clean and optimize content for TTS conversion.
Chapter detection and splitting ensures proper audiobook structure.
"""

    test_file.write_text(test_content)
    print(f'📝 Created test file: {test_file}')

    # Test analysis
    try:
        analysis = converter.analyze_ebook(str(test_file))
        print('📊 Analysis completed:')
        print(f'  - Words: {analysis["total_words"]}')
        print(f'  - Chapters: {analysis["total_chapters"]}')
        print(f'  - Duration: {analysis["estimated_duration_minutes"]:.1f} min')
        
        for chapter in analysis['chapters']:
            print(f'  - Chapter {chapter["number"]}: {chapter["title"]} ({chapter["word_count"]} words)')
            
    except Exception as e:
        print(f'❌ Analysis failed: {e}')
        import traceback
        traceback.print_exc()

    # Cleanup
    test_file.unlink()
    print('✅ Test completed')

if __name__ == "__main__":
    test_converter()

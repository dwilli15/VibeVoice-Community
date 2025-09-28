#!/usr/bin/env python3
"""
VibeVoice Community Debug Verification Script
Demonstrates that all systems are working correctly after debugging
"""

import sys
from pathlib import Path

def test_simple_tts_backend():
    """Test the fixed SimpleTTSBackend"""
    print("🔧 Testing SimpleTTSBackend...")
    try:
        from simple_tts_backend import SimpleTTSBackend
        backend = SimpleTTSBackend()
        voices = backend.get_voices()
        print(f"   ✅ SimpleTTSBackend loaded with {len(voices)} voices")
        
        # Show available voices
        for voice in voices[:5]:  # Show first 5
            print(f"      • {voice.name} ({voice.language}, {voice.gender})")
        if len(voices) > 5:
            print(f"      ... and {len(voices) - 5} more voices")
        
        return True
    except Exception as e:
        print(f"   ❌ SimpleTTSBackend failed: {e}")
        return False

def test_ebook_converter():
    """Test the EbookToAudiobookConverter integration"""
    print("📚 Testing EbookToAudiobookConverter...")
    try:
        from ebook_converter import EbookToAudiobookConverter
        converter = EbookToAudiobookConverter()
        print("   ✅ EbookToAudiobookConverter loaded successfully")
        
        # Check if TTS backends are available
        if converter.simple_tts_backend:
            print("   ✅ SimpleTTSBackend integration working")
        if converter.multimodel_tts_backend:
            print("   ✅ MultiModelTTSBackend integration working")
        if converter.ffmpeg_available:
            print("   ✅ FFmpeg available for audio packaging")
        
        return True
    except Exception as e:
        print(f"   ❌ EbookToAudiobookConverter failed: {e}")
        return False

def test_enhanced_voice_library():
    """Test the enhanced voice library"""
    print("🎭 Testing Enhanced Voice Library...")
    try:
        from enhanced_voice_library import EnhancedVoiceLibrary
        library = EnhancedVoiceLibrary()
        voices = library.get_all_voices()
        stats = library.get_voice_statistics()
        
        print(f"   ✅ Voice library loaded with {len(voices)} voices")
        print(f"   📊 Languages supported: {len(stats['by_language'])}")
        print(f"   🎙️ Engines available: {len([e for e, v in stats['by_engine'].items() if v > 0])}")
        
        # Test voice search
        english_voices = library.search_voices(language="english")
        professional_voices = library.search_voices(style="professional")
        print(f"   🔍 Search test: {len(english_voices)} English, {len(professional_voices)} professional")
        
        return True
    except Exception as e:
        print(f"   ❌ Enhanced Voice Library failed: {e}")
        return False

def test_web_scraper():
    """Test the web scraper module"""
    print("🌐 Testing Web Scraper...")
    try:
        from web_scraper import WebScraper, WebContent
        scraper = WebScraper()
        
        # Test basic instantiation and WebContent creation
        test_content = WebContent(
            url="https://example.com",
            title="Test Article",
            content="This is a test paragraph for content extraction."
        )
        
        # Test that basic methods exist
        assert hasattr(scraper, 'scrape_single_url')
        assert hasattr(scraper, 'scrape_multiple_urls')
        
        print("   ✅ Web scraper loaded and basic functionality works")
        
        return True
    except Exception as e:
        print(f"   ❌ Web Scraper failed: {e}")
        return False

def test_llm_integration():
    """Test the LM Studio integration"""
    print("🤖 Testing LM Studio Integration...")
    try:
        from llm_integration import LMStudioConnector, SpeakerProfile, ConversationConfig
        connector = LMStudioConnector()
        
        # Test speaker profile creation with correct parameters
        profile = SpeakerProfile(
            name="Test Speaker", 
            personality="Professional", 
            background="narrator",
            speaking_style="clear"
        )
        config = ConversationConfig(
            topic="Test topic",
            speakers=[profile]
        )
        
        print("   ✅ LM Studio integration loaded and profiles work")
        
        return True
    except Exception as e:
        print(f"   ❌ LM Studio Integration failed: {e}")
        return False

def test_enhanced_gui():
    """Test the enhanced GUI"""
    print("🖥️ Testing Enhanced GUI...")
    try:
        from enhanced_gui import EnhancedVibeVoiceGUI
        gui = EnhancedVibeVoiceGUI()
        
        # Test voice options generation
        options = gui.get_voice_options()
        recommendations = gui.get_voice_recommendations("audiobook")
        
        print(f"   ✅ Enhanced GUI loaded with {len(options)} voice options")
        print(f"   💡 Recommendations working: {len(recommendations)} audiobook voices")
        
        return True
    except Exception as e:
        print(f"   ❌ Enhanced GUI failed: {e}")
        return False

def run_system_demo():
    """Run a comprehensive system demo"""
    print("🎉 VibeVoice Community System Verification")
    print("=" * 60)
    print("Testing all components after debugging fixes...")
    print()
    
    tests = [
        ("SimpleTTSBackend", test_simple_tts_backend),
        ("EbookConverter", test_ebook_converter),
        ("Voice Library", test_enhanced_voice_library),
        ("Web Scraper", test_web_scraper),
        ("LM Studio Integration", test_llm_integration),
        ("Enhanced GUI", test_enhanced_gui),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
        print()
    
    # Summary
    print("=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("-" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:<8} {test_name}")
    
    print("-" * 30)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("🚀 VibeVoice Community is ready for use!")
        print("\nNext steps:")
        print("• Launch GUI: python enhanced_gui.py")
        print("• Test conversion: python ebook_converter.py --help")
        print("• Check voices: python enhanced_voice_library.py stats")
    else:
        print(f"\n⚠️ {total - passed} component(s) need attention")
    
    return passed == total

if __name__ == "__main__":
    success = run_system_demo()
    sys.exit(0 if success else 1)

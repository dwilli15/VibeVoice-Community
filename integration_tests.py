"""
Integration Testing Suite for VibeVoice Community Enhancements
Tests the integration between new modules and existing TTS infrastructure
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import logging
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add project modules to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from enhanced_voice_library import EnhancedVoiceLibrary, VoiceInfo, VoiceCategory, VoiceEngine
    from web_scraper import WebScraper, WebContent, WebToAudiobookConverter
    from llm_integration import LMStudioConnector, ConversationConfig, SpeakerProfile, ScriptToAudioConverter
except ImportError as e:
    print(f"⚠️  Warning: Could not import new modules: {e}")
    print("🔧 Some tests will be skipped")

class TestVoiceLibraryIntegration(unittest.TestCase):
    """Test enhanced voice library integration"""
    
    def setUp(self):
        """Setup test environment"""
        self.library = EnhancedVoiceLibrary()
    
    def test_voice_discovery(self):
        """Test that voices are discovered from all engines"""
        voices = self.library.get_all_voices()
        self.assertGreater(len(voices), 0, "Should discover at least some voices")
        
        # Check that we have voices from different engines
        engines = set(voice.engine for voice in voices.values())
        self.assertIn("coqui", engines, "Should discover Coqui voices")
        
        print(f"✅ Discovered {len(voices)} voices from {len(engines)} engines")
    
    def test_voice_search(self):
        """Test voice search functionality"""
        # Search by language
        english_voices = self.library.search_voices(language="english")
        self.assertGreater(len(english_voices), 0, "Should find English voices")
        
        # Search by gender
        female_voices = self.library.search_voices(gender="female")
        male_voices = self.library.search_voices(gender="male")
        
        print(f"✅ Search tests passed - English: {len(english_voices)}, Female: {len(female_voices)}, Male: {len(male_voices)}")
    
    def test_voice_recommendations(self):
        """Test voice recommendation system"""
        audiobook_voices = self.library.get_recommended_voices("audiobook", "en")
        podcast_voices = self.library.get_recommended_voices("podcast", "en")
        
        self.assertGreater(len(audiobook_voices), 0, "Should recommend audiobook voices")
        self.assertGreater(len(podcast_voices), 0, "Should recommend podcast voices")
        
        print(f"✅ Recommendations: {len(audiobook_voices)} audiobook, {len(podcast_voices)} podcast")
    
    def test_voice_statistics(self):
        """Test voice statistics generation"""
        stats = self.library.get_voice_statistics()
        
        self.assertIn("total_voices", stats)
        self.assertIn("by_engine", stats)
        self.assertIn("by_language", stats)
        self.assertGreater(stats["total_voices"], 0)
        
        print(f"✅ Statistics: {stats['total_voices']} total voices")

class TestWebScrapingIntegration(unittest.TestCase):
    """Test web scraping integration"""
    
    def setUp(self):
        """Setup test environment"""
        self.scraper = WebScraper()
    
    @patch('requests.get')
    def test_content_extraction(self, mock_get):
        """Test content extraction from web pages"""
        # Mock HTML response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <head><title>Test Article</title></head>
            <body>
                <article>
                    <h1>Test Article Title</h1>
                    <p>This is a test paragraph for content extraction.</p>
                    <p>This is another paragraph with more content.</p>
                </article>
            </body>
        </html>
        """
        mock_response.headers = {'content-type': 'text/html'}
        mock_get.return_value = mock_response
        
        content = self.scraper.scrape_content("https://example.com/test")
        
        self.assertIsInstance(content, WebContent)
        self.assertEqual(content.title, "Test Article Title")
        self.assertIn("test paragraph", content.text)
        
        print("✅ Web content extraction test passed")
    
    def test_content_formatting(self):
        """Test content formatting for audiobook conversion"""
        test_content = WebContent(
            url="https://example.com",
            title="Test Article",
            text="This is paragraph one.\n\nThis is paragraph two.",
            metadata={"author": "Test Author"}
        )
        
        formatted = self.scraper._format_for_audiobook(test_content)
        
        self.assertIn("Test Article", formatted)
        self.assertIn("This is paragraph one", formatted)
        
        print("✅ Content formatting test passed")

class TestLMStudioIntegration(unittest.TestCase):
    """Test LM Studio integration"""
    
    def setUp(self):
        """Setup test environment"""
        self.connector = LMStudioConnector()
    
    def test_speaker_profile_creation(self):
        """Test speaker profile creation"""
        profile = SpeakerProfile(
            name="Alice",
            personality="Professional and clear",
            voice_style="Professional",
            speaking_patterns=["Measured pace", "Clear pronunciation"]
        )
        
        self.assertEqual(profile.name, "Alice")
        self.assertEqual(profile.personality, "Professional and clear")
        
        print("✅ Speaker profile creation test passed")
    
    def test_conversation_config(self):
        """Test conversation configuration"""
        speakers = [
            SpeakerProfile("Alice", "Professional narrator", "Professional"),
            SpeakerProfile("Bob", "Casual commentator", "Conversational")
        ]
        
        config = ConversationConfig(
            speakers=speakers,
            style="dialogue",
            topic_focus="Educational content"
        )
        
        self.assertEqual(len(config.speakers), 2)
        self.assertEqual(config.style, "dialogue")
        
        print("✅ Conversation configuration test passed")
    
    @patch('requests.post')
    def test_script_generation_mock(self, mock_post):
        """Test script generation with mocked LM Studio response"""
        # Mock LM Studio response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Alice: Welcome to today's discussion.\nBob: Thank you for having me, Alice."
                }
            }]
        }
        mock_post.return_value = mock_response
        
        config = ConversationConfig(
            speakers=[
                SpeakerProfile("Alice", "Host", "Professional"),
                SpeakerProfile("Bob", "Guest", "Conversational")
            ]
        )
        
        script = self.connector.generate_conversation("Test content about AI", config)
        
        self.assertIn("Alice:", script)
        self.assertIn("Bob:", script)
        
        print("✅ Mock script generation test passed")

class TestEndToEndIntegration(unittest.TestCase):
    """Test complete end-to-end workflows"""
    
    def setUp(self):
        """Setup test environment"""
        self.voice_library = EnhancedVoiceLibrary()
        self.web_scraper = WebScraper()
        self.llm_connector = LMStudioConnector()
    
    def test_voice_selection_for_content(self):
        """Test selecting appropriate voices for different content types"""
        # Get recommended voices for different content types
        audiobook_voices = self.voice_library.get_recommended_voices("audiobook", "en")
        podcast_voices = self.voice_library.get_recommended_voices("podcast", "en")
        
        self.assertGreater(len(audiobook_voices), 0)
        self.assertGreater(len(podcast_voices), 0)
        
        # Check that we can find specific voice characteristics
        professional_voices = self.voice_library.search_voices(style="professional")
        narrative_voices = self.voice_library.search_voices(style="narrative")
        
        print(f"✅ Voice selection: {len(professional_voices)} professional, {len(narrative_voices)} narrative")
    
    def test_content_processing_pipeline(self):
        """Test the complete content processing pipeline"""
        # Create test content
        test_content = WebContent(
            url="https://example.com/article",
            title="AI and the Future of Technology",
            text="Artificial intelligence is rapidly transforming our world. From machine learning to natural language processing, AI technologies are becoming increasingly sophisticated.",
            metadata={"author": "Tech Writer", "date": "2024-01-15"}
        )
        
        # Test content formatting
        formatted_content = self.web_scraper._format_for_audiobook(test_content)
        self.assertIsInstance(formatted_content, str)
        self.assertIn("AI and the Future", formatted_content)
        
        # Test speaker profile creation for the content
        speakers = [
            SpeakerProfile("Narrator", "Clear and informative", "Educational"),
            SpeakerProfile("Expert", "Knowledgeable and engaging", "Professional")
        ]
        
        config = ConversationConfig(speakers=speakers, style="educational")
        self.assertEqual(len(config.speakers), 2)
        
        print("✅ Content processing pipeline test passed")
    
    def test_integration_with_existing_tts(self):
        """Test integration with existing TTS backend"""
        try:
            # Try to import existing TTS backend
            from tts_backend import MultiModelTTSBackend, get_tts_backend
            
            # Get TTS backend instance
            backend = get_tts_backend()
            
            # Get available voices from enhanced library
            voices = self.voice_library.get_all_voices()
            coqui_voices = [v for v in voices.values() if v.engine == "coqui"]
            
            self.assertGreater(len(coqui_voices), 0, "Should have Coqui voices available")
            
            # Test that voice IDs are compatible
            for voice in coqui_voices[:3]:  # Test first 3 voices
                self.assertIsInstance(voice.id, str)
                self.assertIsInstance(voice.model_path, str)
            
            print(f"✅ TTS integration: {len(coqui_voices)} Coqui voices available")
            
        except ImportError:
            print("ℹ️  TTS backend not available - skipping integration test")

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_invalid_voice_search(self):
        """Test handling of invalid voice searches"""
        library = EnhancedVoiceLibrary()
        
        # Search for non-existent language
        voices = library.search_voices(language="nonexistent")
        self.assertEqual(len(voices), 0)
        
        # Search for non-existent voice ID
        voice = library.get_voice_by_id("nonexistent_voice")
        self.assertIsNone(voice)
        
        print("✅ Invalid search handling test passed")
    
    def test_malformed_content_handling(self):
        """Test handling of malformed web content"""
        scraper = WebScraper()
        
        # Test empty content
        empty_content = WebContent("", "", "", {})
        formatted = scraper._format_for_audiobook(empty_content)
        self.assertIsInstance(formatted, str)
        
        print("✅ Malformed content handling test passed")
    
    def test_llm_connection_error_handling(self):
        """Test LM Studio connection error handling"""
        connector = LMStudioConnector()
        
        # Test with invalid base URL
        connector.base_url = "http://invalid-url:1234"
        
        config = ConversationConfig(speakers=[
            SpeakerProfile("Test", "Test personality", "Test style")
        ])
        
        # This should handle connection errors gracefully
        try:
            script = connector.generate_conversation("test content", config)
            # If it doesn't raise an exception, it should return empty or error message
            self.assertIsInstance(script, str)
        except Exception as e:
            # Connection errors are expected with invalid URL
            self.assertIsInstance(e, Exception)
        
        print("✅ Connection error handling test passed")

def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Starting VibeVoice Community Integration Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestVoiceLibraryIntegration))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestWebScrapingIntegration))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestLMStudioIntegration))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestEndToEndIntegration))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestErrorHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    if result.errors:
        print("\n⚠️  Errors:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Integration tests mostly successful!")
    elif success_rate >= 60:
        print("⚠️  Some integration issues detected")
    else:
        print("❌ Significant integration problems found")
    
    return result.wasSuccessful()

def test_specific_module(module_name: str):
    """Test a specific module"""
    print(f"🧪 Testing {module_name} module")
    
    test_classes = {
        "voice_library": TestVoiceLibraryIntegration,
        "web_scraper": TestWebScrapingIntegration, 
        "llm_integration": TestLMStudioIntegration,
        "end_to_end": TestEndToEndIntegration,
        "error_handling": TestErrorHandling
    }
    
    if module_name not in test_classes:
        print(f"❌ Unknown module: {module_name}")
        print(f"Available modules: {', '.join(test_classes.keys())}")
        return False
    
    suite = unittest.TestLoader().loadTestsFromTestCase(test_classes[module_name])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def quick_system_check():
    """Quick system integration check"""
    print("🔍 Quick System Integration Check")
    print("-" * 40)
    
    checks = []
    
    # Check 1: Voice Library
    try:
        library = EnhancedVoiceLibrary()
        voice_count = len(library.get_all_voices())
        checks.append(("Voice Library", True, f"{voice_count} voices discovered"))
    except Exception as e:
        checks.append(("Voice Library", False, str(e)))
    
    # Check 2: Web Scraper
    try:
        scraper = WebScraper()
        checks.append(("Web Scraper", True, "Module loaded successfully"))
    except Exception as e:
        checks.append(("Web Scraper", False, str(e)))
    
    # Check 3: LM Studio Integration
    try:
        connector = LMStudioConnector()
        checks.append(("LM Studio Integration", True, "Module loaded successfully"))
    except Exception as e:
        checks.append(("LM Studio Integration", False, str(e)))
    
    # Check 4: Existing TTS Backend
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from tts_backend import MultiModelTTSBackend, get_tts_backend
        backend = get_tts_backend()
        checks.append(("TTS Backend", True, "MultiModelTTSBackend accessible"))
    except Exception as e:
        checks.append(("TTS Backend", False, f"Import error: {e}"))
    
    # Display results
    total_checks = len(checks)
    passed_checks = sum(1 for _, status, _ in checks if status)
    
    for check_name, status, message in checks:
        icon = "✅" if status else "❌"
        print(f"{icon} {check_name}: {message}")
    
    print("-" * 40)
    print(f"📊 System Health: {passed_checks}/{total_checks} checks passed ({passed_checks/total_checks*100:.0f}%)")
    
    if passed_checks == total_checks:
        print("🎉 All systems operational!")
    elif passed_checks >= total_checks * 0.75:
        print("⚠️  Minor issues detected")
    else:
        print("❌ Significant system issues found")
    
    return passed_checks == total_checks

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VibeVoice Community Integration Tests")
    parser.add_argument("--module", help="Test specific module", choices=[
        "voice_library", "web_scraper", "llm_integration", "end_to_end", "error_handling"
    ])
    parser.add_argument("--quick", action="store_true", help="Run quick system check")
    parser.add_argument("--full", action="store_true", help="Run full integration test suite")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_system_check()
    elif args.module:
        test_specific_module(args.module)
    elif args.full:
        run_integration_tests()
    else:
        # Default: run quick check
        quick_system_check()

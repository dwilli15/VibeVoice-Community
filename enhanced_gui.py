"""
GUI Integration Module for VibeVoice Community
Integrates new capabilities into existing Gradio interface
"""

import gradio as gr
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

try:
    from enhanced_voice_library import EnhancedVoiceLibrary, VoiceInfo, VoiceCategory
    from web_scraper import WebScraper, ScrapingConfig
    from llm_integration import LMStudioConnector, ConversationConfig, SpeakerProfile
except ImportError as e:
    print(f"⚠️  Warning: Could not import enhancement modules: {e}")

class EnhancedVibeVoiceGUI:
    """Enhanced GUI with new VibeVoice Community features"""
    
    def __init__(self):
        self.voice_library = EnhancedVoiceLibrary()
        self.web_scraper = WebScraper()
        self.llm_connector = LMStudioConnector()
        self.logger = self._setup_logging()
        
        # Cache for expensive operations
        self._voice_cache = {}
        self._recommendations_cache = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('VibeVoiceGUI')
        logger.setLevel(logging.INFO)
        return logger
    
    def get_voice_options(self, language_filter: str = "", 
                         gender_filter: str = "", 
                         style_filter: str = "") -> List[str]:
        """Get voice options for dropdown"""
        voices = self.voice_library.search_voices(
            language=language_filter,
            gender=gender_filter, 
            style=style_filter
        )
        
        voice_options = []
        for voice in voices:
            display_name = f"{voice.name} ({voice.language}, {voice.gender}, {voice.quality})"
            voice_options.append(display_name)
        
        return voice_options
    
    def get_voice_recommendations(self, content_type: str, language: str = "en") -> List[str]:
        """Get voice recommendations for content type"""
        cache_key = f"{content_type}_{language}"
        
        if cache_key not in self._recommendations_cache:
            voices = self.voice_library.get_recommended_voices(content_type, language)
            recommendations = []
            
            for voice in voices[:10]:  # Top 10 recommendations
                display_name = f"{voice.name} ({voice.style}, {voice.quality})"
                recommendations.append(display_name)
            
            self._recommendations_cache[cache_key] = recommendations
        
        return self._recommendations_cache[cache_key]
    
    def process_web_content(self, urls: str, conversion_format: str = "audiobook") -> Tuple[str, str]:
        """Process web content for TTS conversion"""
        try:
            # Parse URLs
            url_list = [url.strip() for url in urls.split('\n') if url.strip()]
            
            if not url_list:
                return "❌ Error: No valid URLs provided", ""
            
            # Configure scraping
            config = ScrapingConfig(
                respect_robots_txt=True,
                delay_between_requests=1.0,
                max_content_length=50000,
                output_format=conversion_format
            )
            
            # Scrape content
            all_content = []
            for url in url_list:
                try:
                    content = self.web_scraper.scrape_content(url, config)
                    if content:
                        formatted = self.web_scraper._format_for_audiobook(content)
                        all_content.append(formatted)
                except Exception as e:
                    self.logger.warning(f"Failed to scrape {url}: {e}")
                    continue
            
            if not all_content:
                return "❌ Error: No content could be extracted from provided URLs", ""
            
            # Combine content
            combined_content = "\n\n---\n\n".join(all_content)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(combined_content)
                temp_file = f.name
            
            success_msg = f"✅ Successfully processed {len(all_content)} pages from {len(url_list)} URLs"
            return success_msg, temp_file
            
        except Exception as e:
            return f"❌ Error processing web content: {str(e)}", ""
    
    def generate_multi_speaker_script(self, content: str, num_speakers: int = 2, 
                                     content_style: str = "dialogue") -> Tuple[str, str]:
        """Generate multi-speaker script using LM Studio"""
        try:
            if not content.strip():
                return "❌ Error: No content provided", ""
            
            # Create speaker profiles based on content style
            speakers = self._create_speaker_profiles(num_speakers, content_style)
            
            # Configure conversation
            config = ConversationConfig(
                speakers=speakers,
                style=content_style,
                max_length=2000,
                topic_focus="Content adaptation"
            )
            
            # Generate script (will gracefully handle LM Studio unavailability)
            script = self.llm_connector.generate_conversation(content, config)
            
            if not script or script.startswith("Error"):
                # Fallback: create simple script format
                script = self._create_fallback_script(content, speakers)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(script)
                temp_file = f.name
            
            success_msg = f"✅ Generated {num_speakers}-speaker script ({len(script)} characters)"
            return success_msg, temp_file
            
        except Exception as e:
            return f"❌ Error generating script: {str(e)}", ""
    
    def _create_speaker_profiles(self, num_speakers: int, style: str) -> List[SpeakerProfile]:
        """Create speaker profiles based on content style"""
        style_templates = {
            "dialogue": [
                ("Host", "Professional and engaging interviewer", "Professional"),
                ("Guest", "Knowledgeable expert or participant", "Conversational"),
                ("Moderator", "Neutral facilitator", "Clear"),
                ("Commentator", "Analytical and insightful", "Thoughtful")
            ],
            "educational": [
                ("Teacher", "Clear and instructional", "Educational"),
                ("Student", "Curious and questioning", "Young"),
                ("Expert", "Authoritative and detailed", "Professional"),
                ("Assistant", "Helpful and supportive", "Friendly")
            ],
            "storytelling": [
                ("Narrator", "Engaging storyteller", "Narrative"),
                ("Character1", "Dynamic character voice", "Character"),
                ("Character2", "Contrasting character voice", "Character"),
                ("Chorus", "Collective voice", "Group")
            ],
            "podcast": [
                ("Host", "Charismatic podcast host", "Engaging"),
                ("CoHost", "Complementary co-host", "Friendly"),
                ("Expert", "Subject matter expert", "Authoritative"),
                ("Caller", "Audience representative", "Casual")
            ]
        }
        
        templates = style_templates.get(style, style_templates["dialogue"])
        speakers = []
        
        for i in range(min(num_speakers, len(templates))):
            name, personality, voice_style = templates[i]
            speakers.append(SpeakerProfile(
                name=name,
                personality=personality,
                voice_style=voice_style
            ))
        
        return speakers
    
    def _create_fallback_script(self, content: str, speakers: List[SpeakerProfile]) -> str:
        """Create fallback script when LM Studio is unavailable"""
        # Simple script format with speaker rotation
        sentences = content.split('. ')
        script_lines = []
        
        script_lines.append("# Multi-Speaker Script")
        script_lines.append("# Generated by VibeVoice Community")
        script_lines.append("")
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                speaker = speakers[i % len(speakers)]
                script_lines.append(f"{speaker.name}: {sentence.strip()}.")
                script_lines.append("")
        
        return "\n".join(script_lines)
    
    def get_library_statistics(self) -> str:
        """Get voice library statistics for display"""
        stats = self.voice_library.get_voice_statistics()
        
        stats_text = "📊 **Voice Library Statistics**\n\n"
        stats_text += f"**Total Voices:** {stats['total_voices']}\n\n"
        
        stats_text += "**By Engine:**\n"
        for engine, count in stats['by_engine'].items():
            stats_text += f"- {engine.title()}: {count} voices\n"
        
        stats_text += "\n**By Language (Top 10):**\n"
        sorted_langs = sorted(stats['by_language'].items(), key=lambda x: x[1], reverse=True)
        for lang, count in sorted_langs[:10]:
            stats_text += f"- {lang}: {count} voices\n"
        
        stats_text += "\n**By Quality:**\n"
        for quality, count in stats['by_quality'].items():
            if count > 0:
                stats_text += f"- {quality.title()}: {count} voices\n"
        
        return stats_text
    
    def create_enhanced_interface(self) -> gr.Blocks:
        """Create enhanced Gradio interface"""
        
        with gr.Blocks(
            title="VibeVoice Community - Enhanced TTS Platform",
            theme=gr.themes.Soft(),
            css="""
            .main-header { text-align: center; margin-bottom: 2rem; }
            .feature-section { margin: 1.5rem 0; padding: 1rem; border-radius: 8px; background-color: #f8f9fa; }
            .stats-display { font-family: monospace; white-space: pre-wrap; }
            """
        ) as interface:
            
            # Header
            gr.Markdown(
                """
                # 🎙️ VibeVoice Community - Enhanced TTS Platform
                
                Transform any content into professional audio with AI-powered voice synthesis, 
                web scraping, and multi-speaker script generation.
                """,
                elem_classes=["main-header"]
            )
            
            with gr.Tabs():
                
                # Voice Library Tab
                with gr.TabItem("🎭 Voice Library"):
                    gr.Markdown("### Explore 65+ Professional Voices")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            language_filter = gr.Dropdown(
                                choices=[""] + list(self.voice_library.languages.keys()),
                                label="Filter by Language",
                                value=""
                            )
                            gender_filter = gr.Dropdown(
                                choices=["", "male", "female", "mixed", "neutral"],
                                label="Filter by Gender",
                                value=""
                            )
                            style_filter = gr.Dropdown(
                                choices=["", "professional", "conversational", "narrative", "educational"],
                                label="Filter by Style", 
                                value=""
                            )
                            
                            search_voices_btn = gr.Button("🔍 Search Voices", variant="primary")
                        
                        with gr.Column(scale=2):
                            voice_results = gr.Dataframe(
                                headers=["Voice Name", "Language", "Gender", "Style", "Quality", "Engine"],
                                label="Available Voices",
                                interactive=False
                            )
                    
                    with gr.Row():
                        stats_display = gr.Markdown(
                            value=self.get_library_statistics(),
                            elem_classes=["stats-display"]
                        )
                
                # Web Content Processing Tab
                with gr.TabItem("🌐 Web Content"):
                    gr.Markdown("### Convert Web Content to Audio")
                    
                    with gr.Row():
                        with gr.Column():
                            urls_input = gr.Textbox(
                                lines=5,
                                placeholder="Enter URLs (one per line):\nhttps://example.com/article1\nhttps://example.com/article2",
                                label="Web URLs to Process"
                            )
                            
                            content_type_web = gr.Dropdown(
                                choices=["audiobook", "podcast", "news", "educational"],
                                value="audiobook",
                                label="Content Format"
                            )
                            
                            process_web_btn = gr.Button("🕷️ Process Web Content", variant="primary")
                        
                        with gr.Column():
                            web_status = gr.Textbox(label="Processing Status", interactive=False)
                            web_content_file = gr.File(label="Processed Content", interactive=False)
                
                # Multi-Speaker Script Generation Tab  
                with gr.TabItem("🎬 Multi-Speaker Scripts"):
                    gr.Markdown("### Generate AI-Powered Multi-Speaker Scripts")
                    
                    with gr.Row():
                        with gr.Column():
                            script_content = gr.Textbox(
                                lines=8,
                                placeholder="Enter your content here. The AI will convert it into a multi-speaker dialogue format suitable for TTS generation.",
                                label="Content to Convert"
                            )
                            
                            with gr.Row():
                                num_speakers = gr.Slider(
                                    minimum=2, maximum=4, value=2, step=1,
                                    label="Number of Speakers"
                                )
                                script_style = gr.Dropdown(
                                    choices=["dialogue", "educational", "storytelling", "podcast"],
                                    value="dialogue",
                                    label="Script Style"
                                )
                            
                            generate_script_btn = gr.Button("🤖 Generate Script", variant="primary")
                        
                        with gr.Column():
                            script_status = gr.Textbox(label="Generation Status", interactive=False)
                            script_file = gr.File(label="Generated Script", interactive=False)
                
                # Voice Recommendations Tab
                with gr.TabItem("💡 Recommendations"):
                    gr.Markdown("### Get AI-Powered Voice Recommendations")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            rec_content_type = gr.Dropdown(
                                choices=["audiobook", "podcast", "news", "educational", "entertainment", "general"],
                                value="general",
                                label="Content Type"
                            )
                            rec_language = gr.Dropdown(
                                choices=["en", "es", "fr", "de", "it", "pt", "zh", "ja"],
                                value="en",
                                label="Primary Language"
                            )
                            
                            get_recommendations_btn = gr.Button("🎯 Get Recommendations", variant="primary")
                        
                        with gr.Column(scale=2):
                            recommendations_display = gr.Markdown(
                                value="Click 'Get Recommendations' to see AI-powered voice suggestions for your content type.",
                                label="Recommended Voices"
                            )
                
                # Integration Status Tab
                with gr.TabItem("🔧 System Status"):
                    gr.Markdown("### System Integration Status")
                    
                    with gr.Column():
                        refresh_status_btn = gr.Button("🔄 Refresh System Status")
                        
                        system_status = gr.Markdown(
                            value="""
                            **System Status:** Initializing...
                            
                            Click 'Refresh System Status' to check all systems.
                            """,
                            elem_classes=["stats-display"]
                        )
            
            # Event handlers
            def search_voices_handler(language, gender, style):
                voices = self.voice_library.search_voices(
                    language=language, gender=gender, style=style
                )
                
                # Convert to dataframe format
                voice_data = []
                for voice in voices[:50]:  # Limit to 50 results
                    voice_data.append([
                        voice.name,
                        voice.language,
                        voice.gender,
                        voice.style,
                        voice.quality,
                        voice.engine
                    ])
                
                return voice_data
            
            def get_recommendations_handler(content_type, language):
                recommendations = self.get_voice_recommendations(content_type, language)
                
                if not recommendations:
                    return "No recommendations available for this combination."
                
                rec_text = f"## 🎯 Recommended Voices for {content_type.title()} Content\n\n"
                for i, rec in enumerate(recommendations, 1):
                    rec_text += f"{i}. **{rec}**\n"
                
                return rec_text
            
            def refresh_status_handler():
                try:
                    # Run quick system check
                    from integration_tests import quick_system_check
                    
                    # Capture system check output
                    import io
                    import contextlib
                    
                    f = io.StringIO()
                    with contextlib.redirect_stdout(f):
                        quick_system_check()
                    
                    status_output = f.getvalue()
                    return f"```\n{status_output}\n```"
                    
                except Exception as e:
                    return f"❌ Error checking system status: {str(e)}"
            
            # Connect event handlers
            search_voices_btn.click(
                search_voices_handler,
                inputs=[language_filter, gender_filter, style_filter],
                outputs=[voice_results]
            )
            
            process_web_btn.click(
                self.process_web_content,
                inputs=[urls_input, content_type_web],
                outputs=[web_status, web_content_file]
            )
            
            generate_script_btn.click(
                self.generate_multi_speaker_script,
                inputs=[script_content, num_speakers, script_style],
                outputs=[script_status, script_file]
            )
            
            get_recommendations_btn.click(
                get_recommendations_handler,
                inputs=[rec_content_type, rec_language],
                outputs=[recommendations_display]
            )
            
            refresh_status_btn.click(
                refresh_status_handler,
                inputs=[],
                outputs=[system_status]
            )
        
        return interface

def launch_enhanced_gui():
    """Launch the enhanced VibeVoice GUI"""
    print("🚀 Launching Enhanced VibeVoice Community GUI...")
    
    try:
        gui = EnhancedVibeVoiceGUI()
        interface = gui.create_enhanced_interface()
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    launch_enhanced_gui()

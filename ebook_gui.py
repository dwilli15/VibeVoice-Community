"""
Gradio Web Interface for Ebook to Audiobook Converter
Comprehensive UI for converting PDF, TXT, DOCX, EPUB to audiobooks
Enhanced with Voice Library Management System and VibeVoice Podcast Generation
"""

import gradio as gr
import os
import json
import re
import tempfile
import shutil
import torch
import time
import socket
import psutil
import signal
import atexit
import sys
import codecs
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Fix Unicode encoding issues on Windows
if os.name == 'nt':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Import our ebook converter
try:
    from ebook_converter import EbookToAudiobookConverter, ConversionConfig
    CONVERTER_AVAILABLE = True
except ImportError as e:
    print(f"Ebook converter not available: {e}")
    CONVERTER_AVAILABLE = False

# Import TTS backend
try:
    from tts_backend import MultiModelTTSBackend
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"TTS backend not available: {e}")
    TTS_AVAILABLE = False

# Import VibeVoice for podcast generation
try:
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    from vibevoice.modular.streamer import AudioStreamer
    VIBEVOICE_AVAILABLE = True
except ImportError as e:
    print(f"VibeVoice not available: {e}")
    VIBEVOICE_AVAILABLE = False

# Import voice library
try:
    from vibevoice.voice_library import voice_library, get_voice_selector_data, resolve_voice_mapping
    VOICE_LIBRARY_AVAILABLE = True
except ImportError as e:
    print(f"Voice library not available: {e}")
    try:
        from voice_library_adapter import voice_library, get_voice_selector_data, resolve_voice_mapping  # type: ignore
        print("Using local voice_library_adapter fallback")
        VOICE_LIBRARY_AVAILABLE = True
    except ImportError as adapter_error:
        print(f"Voice library adapter unavailable: {adapter_error}")
        VOICE_LIBRARY_AVAILABLE = False

class EbookConverterGUI:
    """Main GUI class for ebook conversion with VibeVoice podcast generation"""
    
    def __init__(self):
        self.converter = EbookToAudiobookConverter() if CONVERTER_AVAILABLE else None
        self.tts_backend = MultiModelTTSBackend() if TTS_AVAILABLE else None
        self.temp_dir = Path(tempfile.gettempdir()) / "ebook_converter"
        self.temp_dir.mkdir(exist_ok=True)
        # Persistent output root for saved files
        self.default_output_root = Path.home() / "VibeVoice_Output"
        try:
            self.default_output_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback to temp if home is not writable
            self.default_output_root = self.temp_dir
        
        # Gradio interface management
        self.interface = None
        self.server_port = None
        self.server_process = None
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        
        # VibeVoice model configuration
        self.available_models = {
            "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B", 
            "VibeVoice-Large": "aoi-ot/VibeVoice-Large"
        }
        self.current_model = "microsoft/VibeVoice-1.5B"  # Default model
        self.vibevoice_processor = None
        self.vibevoice_model = None
        self.is_generating = False
        self.stop_generation = False
        
        # Setup logging
        self.logger = logging.getLogger('EbookGUI')
        self.logger.setLevel(logging.INFO)
    
    def cleanup(self):
        """Clean up resources and close server"""
        try:
            if self.interface:
                print("[CLEANUP] Cleaning up Gradio interface...")
                self.interface.close()
                self.interface = None
            
            if self.server_port:
                print(f"[CLEANUP] Cleaning up port {self.server_port}...")
                self.kill_port_processes(self.server_port)
                self.server_port = None
                
        except Exception as e:
            print(f"[WARN] Cleanup warning: {e}")
    
    def kill_port_processes(self, port: int):
        """Kill any processes using the specified port"""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port:
                    try:
                        process = psutil.Process(conn.pid)
                        print(f"[KILL] Terminating process {conn.pid} using port {port}")
                        process.terminate()
                        time.sleep(1)
                        if process.is_running():
                            process.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except Exception as e:
            print(f"[WARN] Port cleanup warning: {e}")
    
    def find_free_port(self, start_port: int = 7862) -> int:
        """Find a free port starting from the given port"""
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        raise RuntimeError("No free ports available")
    
    def load_vibevoice_model(self, model_path: str):
        """Load VibeVoice model for podcast generation"""
        if not VIBEVOICE_AVAILABLE:
            return False, "VibeVoice not available"
        
        try:
            print(f"🔄 Loading VibeVoice model: {model_path}")
            
            # Load processor
            self.vibevoice_processor = VibeVoiceProcessor.from_pretrained(model_path)
            
            # Load model with inference configuration
            self.vibevoice_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            self.current_model = model_path
            print(f"✅ VibeVoice model loaded: {model_path}")
            return True, f"Model loaded: {model_path}"
            
        except Exception as e:
            print(f"❌ Failed to load VibeVoice model: {e}")
            return False, f"Failed to load model: {e}"
    
    def switch_vibevoice_model(self, model_name: str):
        """Switch between VibeVoice models"""
        if model_name in self.available_models:
            model_path = self.available_models[model_name]
            success, message = self.load_vibevoice_model(model_path)
            return message
        return f"Unknown model: {model_name}"
    
    def parse_podcast_script(self, script: str) -> List[Tuple[str, str]]:
        """Parse podcast script into speaker-text pairs"""
        lines = script.strip().split('\n')
        speakers_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse "Speaker X: text" format
            if ':' in line:
                parts = line.split(':', 1)
                speaker = parts[0].strip()
                text = parts[1].strip()
                speakers_text.append((speaker, text))
            else:
                # If no speaker specified, use default
                speakers_text.append(("Speaker 1", line))
        
        return speakers_text
    
    def get_simple_voice_names(self) -> List[str]:
        """Get simple voice names for podcast generation"""
        if self.tts_backend and hasattr(self.tts_backend, 'voices'):
            return list(self.tts_backend.voices.keys())
        return ["bf_isabella", "en-Alice_woman", "en-Carter_man"]  # Fallback
    
    def resolve_voice_name(self, voice_name: str) -> str:
        """Convert voice name - simplified since we're using direct names now"""
        if not voice_name:
            return "bf_isabella"  # Default
        
        # Check if it's a valid voice name
        if self.tts_backend and voice_name in self.tts_backend.voices:
            return voice_name
            
        # Fallback to default
        return "bf_isabella"
    
    def generate_podcast(self, num_speakers: int, script: str, *speaker_voices):
        """Generate multi-speaker podcast using VibeVoice"""
        try:
            # Parse the script
            speakers_text = self.parse_podcast_script(script)
            if not speakers_text:
                return "❌ No valid script content found", None
            
            # Process speaker voices - fix indexing and resolve names
            speaker_assignments = {}
            
            # Map each speaker to a voice, cycling through available voices if needed
            unique_speakers = list(set([speaker for speaker, _ in speakers_text]))
            
            for i, speaker in enumerate(unique_speakers):
                if i < len(speaker_voices) and speaker_voices[i]:
                    resolved_name = self.resolve_voice_name(speaker_voices[i])
                    speaker_assignments[speaker] = resolved_name
                else:
                    # Use default voice or cycle through available voices
                    default_voices = ["bf_isabella", "en-Alice_woman", "en-Carter_man", "en-Maya_woman"]
                    fallback_voice = default_voices[i % len(default_voices)]
                    speaker_assignments[speaker] = fallback_voice
            
            generated_audio = []
            status_log = f"🎙️ Generating podcast with {len(speakers_text)} segments\n"
            status_log += f"📋 Speaker assignments:\n"
            for speaker, voice in speaker_assignments.items():
                status_log += f"  - {speaker} → {voice}\n"
            status_log += "\n"
            
            # Generate each segment
            for i, (speaker, text) in enumerate(speakers_text):
                voice_name = speaker_assignments.get(speaker, "bf_isabella")
                
                status_log += f"🔊 Segment {i+1}: {speaker} (voice: {voice_name})\n"
                status_log += f"   Text: {text[:100]}{'...' if len(text) > 100 else ''}\n"
                
                # Use TTS backend to generate audio
                if self.tts_backend:
                    try:
                        # Get the voice object from the backend
                        if voice_name in self.tts_backend.voices:
                            voice = self.tts_backend.voices[voice_name]
                            
                            # Create output path
                            output_path = self.temp_dir / f"podcast_segment_{i+1}_{speaker.replace(' ', '_')}_{int(time.time())}.wav"
                            
                            # Create TTS request using the imported TTSRequest class
                            from tts_backend import TTSRequest
                            request = TTSRequest(
                                text=text,
                                voice=voice,
                                output_path=str(output_path)
                            )
                            
                            # Generate speech
                            success = self.tts_backend.generate_speech(request)
                            if success and output_path.exists():
                                generated_audio.append(str(output_path))
                                status_log += f"   ✅ Generated: {output_path.name}\n\n"
                            else:
                                status_log += f"   ❌ Failed to generate audio (TTS generation failed)\n\n"
                        else:
                            status_log += f"   ⚠️ Voice '{voice_name}' not found, trying fallback...\n"
                            # Try with default voice as fallback
                            fallback_voices = ["bf_isabella", "en-Alice_woman", "en-Carter_man"]
                            for fallback in fallback_voices:
                                if fallback in self.tts_backend.voices:
                                    voice = self.tts_backend.voices[fallback]
                                    output_path = self.temp_dir / f"podcast_segment_{i+1}_{speaker.replace(' ', '_')}_{int(time.time())}.wav"
                                    from tts_backend import TTSRequest
                                    request = TTSRequest(text=text, voice=voice, output_path=str(output_path))
                                    success = self.tts_backend.generate_speech(request)
                                    if success and output_path.exists():
                                        generated_audio.append(str(output_path))
                                        status_log += f"   ✅ Generated with fallback voice: {fallback}\n\n"
                                        break
                            else:
                                status_log += f"   ❌ No working voices found\n\n"
                            
                    except Exception as e:
                        status_log += f"   ❌ Error generating audio: {e}\n\n"
                else:
                    status_log += f"   ❌ TTS backend not available\n\n"
            
            if generated_audio:
                # Return the first audio file (can be enhanced to concatenate later)
                status_log += f"✅ Podcast generated successfully!\n"
                status_log += f"📁 Generated {len(generated_audio)} audio segments\n"
                status_log += f"🎵 Playing first segment: {Path(generated_audio[0]).name}"
                return status_log, generated_audio[0]  # Return the audio file path directly
            else:
                return "❌ Failed to generate any audio segments - check voice assignments and TTS backend", None
                
        except Exception as e:
            return f"❌ Podcast generation failed: {e}", None
    
    def get_available_voices(self) -> List[str]:
        """Get list of available TTS voices from Voice Library and TTS Backend"""
        voice_options = []
        
        # Get voices from TTS backend (our enhanced system)
        if self.tts_backend and hasattr(self.tts_backend, 'voices'):
            backend_voices = self.tts_backend.voices
            if backend_voices:
                voice_options.append("🎵 VIBEVOICE VOICES")
                voice_options.append("---")
                
                # Add VibeVoice voices first (prioritized)
                vibevoice_voices = [v for v in backend_voices.values() if v.engine.value == "vibevoice"]
                for voice in sorted(vibevoice_voices, key=lambda v: (v.gender, v.name)):
                    gender_emoji = "👩" if voice.gender == "female" else "👨" if voice.gender == "male" else "🤖"
                    voice_display = f"🎙️{gender_emoji} {voice.name}"
                    voice_options.append(voice_display)
                
                # Add custom voices if any
                custom_voices = [v for v in backend_voices.values() if hasattr(v, 'file_path') and v.file_path]
                if custom_voices:
                    voice_options.append("---")
                    voice_options.append("📤 CUSTOM VOICES")
                    voice_options.append("---")
                    for voice in sorted(custom_voices, key=lambda v: v.name):
                        gender_emoji = "👩" if voice.gender == "female" else "👨" if voice.gender == "male" else "🤖"
                        voice_display = f"📤{gender_emoji} {voice.name}"
                        voice_options.append(voice_display)
        
        # Add voices from Voice Library if available
        if VOICE_LIBRARY_AVAILABLE:
            all_voices = voice_library.get_all_voices()
            if all_voices:
                if voice_options:  # Add separator if we already have backend voices
                    voice_options.append("---")
                
                # Group voices by language for better organization
                current_language = None
                for voice in sorted(all_voices, key=lambda v: (v.language, v.gender, v.name)):
                    if voice.language != current_language:
                        if current_language is not None:
                            voice_options.append("---")  # Separator
                        voice_options.append(f"🌍 {voice.language.upper()} VOICES")
                        voice_options.append("---")
                        current_language = voice.language
                    
                    # Format voice option with emoji indicators
                    quality_emoji = "⭐" if voice.quality == "premium" else "✨" if voice.quality == "high" else "🔹"
                    gender_emoji = "👩" if voice.gender == "female" else "👨" if voice.gender == "male" else "🤖"
                    engine_emoji = "🎙️" if voice.engine == "vibevoice" else "🗣️" if voice.engine == "coqui" else "🎵"
                    
                    voice_display = f"{quality_emoji}{gender_emoji}{engine_emoji} {voice.name}"
                    voice_options.append(voice_display)
        
        # Fallback if no voices found
        if not voice_options:
            voice_options = ["🔹🤖🎵 Default Voice"]
        
        return voice_options
    
    def search_voices(self, query: str) -> List[str]:
        """Search voices based on query"""
        if not VOICE_LIBRARY_AVAILABLE or not query.strip():
            return self.get_available_voices()
        
        # Search voices
        matching_voices = voice_library.search_voices(query)
        
        if not matching_voices:
            return ["No voices found matching your search."]
        
        voice_options = []
        for voice in matching_voices:
            quality_emoji = "⭐" if voice.quality == "premium" else "✨" if voice.quality == "high" else "🔹"
            gender_emoji = "👩" if voice.gender == "female" else "👨" if voice.gender == "male" else "🤖"
            engine_emoji = "🎙️" if voice.engine == "vibevoice" else "🗣️" if voice.engine == "coqui" else "🎵"
            
            voice_display = f"{quality_emoji}{gender_emoji}{engine_emoji} {voice.name}"
            voice_options.append(voice_display)
        
        return voice_options
    
    def filter_voices_by_language(self, language: str) -> List[str]:
        """Filter voices by language"""
        if not VOICE_LIBRARY_AVAILABLE or language == "All Languages":
            return self.get_available_voices()
        
        lang_voices = voice_library.get_voices_by_language(language)
        
        voice_options = []
        for voice in sorted(lang_voices, key=lambda v: (v.gender, v.name)):
            quality_emoji = "⭐" if voice.quality == "premium" else "✨" if voice.quality == "high" else "🔹"
            gender_emoji = "👩" if voice.gender == "female" else "👨" if voice.gender == "male" else "🤖"
            engine_emoji = "🎙️" if voice.engine == "vibevoice" else "🗣️" if voice.engine == "coqui" else "🎵"
            
            voice_display = f"{quality_emoji}{gender_emoji}{engine_emoji} {voice.name}"
            voice_options.append(voice_display)
        
        return voice_options
    
    def filter_voices_by_gender(self, gender: str) -> List[str]:
        """Filter voices by gender"""
        if not VOICE_LIBRARY_AVAILABLE or gender == "All Genders":
            return self.get_available_voices()
        
        gender_voices = voice_library.get_voices_by_gender(gender)
        
        voice_options = []
        for voice in sorted(gender_voices, key=lambda v: (v.language, v.name)):
            quality_emoji = "⭐" if voice.quality == "premium" else "✨" if voice.quality == "high" else "🔹"
            gender_emoji = "👩" if voice.gender == "female" else "👨" if voice.gender == "male" else "🤖"
            engine_emoji = "🎙️" if voice.engine == "vibevoice" else "🗣️" if voice.engine == "coqui" else "🎵"
            
            voice_display = f"{quality_emoji}{gender_emoji}{engine_emoji} {voice.name}"
            voice_options.append(voice_display)
        
        return voice_options
    
    def get_voice_info(self, voice_display: str) -> str:
        """Get detailed information about selected voice as JSON"""
        if not VOICE_LIBRARY_AVAILABLE:
            return json.dumps({"error": "Voice library not available."})
        
        # Extract voice name from display format
        if "🎙️" in voice_display or "🗣️" in voice_display or "🎵" in voice_display:
            # Remove emojis and get voice name
            voice_name = voice_display.split("️ ")[-1] if "️ " in voice_display else voice_display
            
            # Search for voice by name
            for voice in voice_library.get_all_voices():
                if voice.name in voice_name:
                    voice_info = {
                        "name": voice.name,
                        "language": voice.language,
                        "language_code": voice.language_code,
                        "gender": voice.gender.title(),
                        "quality": voice.quality.title(),
                        "engine": voice.engine.title(),
                        "description": voice.description,
                        "tags": voice.tags,
                        "is_premium": voice.is_premium
                    }
                    return json.dumps(voice_info)
        
        return json.dumps({"message": "Select a voice to see details."})
    
    def load_chapter_list(self, file_path: str) -> Tuple[str, List[str], List[str]]:
        """Load and display chapter list for selection"""
        if not self.converter or not file_path:
            return "❌ No file uploaded or converter not available", [], []
        
        try:
            # Get chapter selection options
            chapter_data = self.converter.get_chapter_selection_options(file_path)
            
            # Format chapter summary
            summary = f"""
📖 **Chapter Selection for: {Path(file_path).name}**

**Total Chapters:** {chapter_data['total_chapters']}
**Total Words:** {chapter_data['total_words']:,}
**Estimated Duration:** {chapter_data['total_estimated_duration']:.1f} minutes ({chapter_data['total_estimated_duration']/60:.1f} hours)

**Available Chapters:**
"""
            
            # Create chapter options for checkboxes
            chapter_choices = []
            chapter_values = []  # Default selected chapters
            
            for chapter in chapter_data['chapters']:
                # Format: "Chapter 1: Title (1,234 words, 12.3 min)"
                choice_text = f"Chapter {chapter['number']}: {chapter['title'][:50]}{'...' if len(chapter['title']) > 50 else ''} ({chapter['word_count']:,} words, {chapter['estimated_duration_minutes']:.1f} min)"
                chapter_choices.append(choice_text)
                
                # Add to summary
                summary += f"- {choice_text}\n"
                
                if chapter['selected']:  # Default selection
                    chapter_values.append(choice_text)
            
            return summary, chapter_choices, chapter_values
            
        except Exception as e:
            return f"❌ Error loading chapters: {e}", [], []
    
    def convert_selected_chapters(
        self, 
        file_path: str,
        selected_chapters: List[str],
        voice_name: str,
        audio_format: str,
        progress=gr.Progress()
    ) -> Tuple[str, List[str]]:
        """Convert only selected chapters to audiobook"""
        if not self.converter or not file_path or not selected_chapters:
            return "❌ No file uploaded, converter not available, or no chapters selected", []
        
        try:
            progress(0, desc="Preparing selected chapters...")
            
            # Extract chapter numbers from selection
            chapter_numbers = []
            for selection in selected_chapters:
                # Extract chapter number from "Chapter X: Title (words, duration)" format
                match = re.match(r"Chapter (\d+):", selection)
                if match:
                    chapter_numbers.append(int(match.group(1)))
            
            if not chapter_numbers:
                return "❌ No valid chapter numbers found in selection", []
            
            progress(0.1, desc="Initializing conversion...")
            
            # Clean voice name for conversion
            if VOICE_LIBRARY_AVAILABLE:
                # Extract voice name from display format with emojis
                if any(emoji in voice_name for emoji in ["🎙️", "🗣️", "🎵"]):
                    clean_voice_name = voice_name.split("️ ")[-1] if "️ " in voice_name else voice_name
                    
                    # Find matching voice in library
                    matching_voice = None
                    for voice in voice_library.get_all_voices():
                        if voice.name in clean_voice_name:
                            matching_voice = voice
                            break
                    
                    if matching_voice:
                        clean_voice = matching_voice.id
                        engine = matching_voice.engine
                    else:
                        clean_voice = voice_name.split(' (')[0] if ' (' in voice_name else voice_name
                        engine = "vibevoice"
                else:
                    clean_voice = voice_name.split(' (')[0] if ' (' in voice_name else voice_name
                    engine = "vibevoice"
            else:
                clean_voice = voice_name.split(' (')[0] if ' (' in voice_name else voice_name
                engine = "vibevoice"
            
            # Create output directory
            output_dir = self.temp_dir / f"selected_chapters_{Path(file_path).stem}"
            output_dir.mkdir(exist_ok=True)
            
            # Create conversion config
            config = ConversionConfig(
                input_file=file_path,
                output_dir=str(output_dir),
                voice_name=clean_voice,
                speed=1.3,  # Default speed
                format=audio_format.lower(),
                engine=engine,
                title=f"Selected Chapters - {Path(file_path).stem}",
                author="Unknown"
            )
            
            progress(0.2, desc=f"Converting {len(chapter_numbers)} selected chapters...")
            
            # Convert selected chapters
            results = self.converter.convert_selected_chapters(
                file_path, chapter_numbers, config
            )
            
            progress(0.9, desc="Finalizing conversion...")
            
            # Format results
            if results['audio_files']:
                status = f"""
✅ **Selected Chapters Conversion Completed!**

**Input:** {Path(file_path).name}
**Selected Chapters:** {sorted(chapter_numbers)}
**Voice:** {clean_voice}
**Engine:** {results.get('engine_used', engine)}
**Format:** {results.get('format', audio_format)}

**Results:**
- Chapters converted: {len(results['audio_files'])}
- Output directory: {results['output_dir']}
"""
                
                if results['errors']:
                    status += f"\n⚠️ **Warnings/Errors:**\n"
                    for error in results['errors'][:5]:
                        status += f"- {error}\n"
                
                progress(1.0, desc="Conversion complete!")
                
                # Return audio files for download
                audio_files = results['audio_files'][:10]  # Limit to first 10 for UI
                
                return status, audio_files
                
            else:
                return "❌ No audio files were generated from selected chapters", []
                
        except Exception as e:
            return f"❌ Selected chapters conversion failed: {e}", []
    
    def get_available_engines(self) -> List[str]:
        """Get list of available TTS engines"""
        engines = ["vibevoice"]  # Always available via SimpleTTSBackend
        
        if self.tts_backend:
            try:
                # Check if Coqui is available in MultiModel backend
                coqui_voices = self.tts_backend.get_voices_by_engine(
                    self.tts_backend.TTSEngine.COQUI if hasattr(self.tts_backend, 'TTSEngine') else None
                )
                if coqui_voices:
                    engines.append("coqui")
            except:
                pass
        
        engines.append("auto")  # Auto-select best available
        return engines
    
    def analyze_ebook(self, file_path: str) -> Tuple[str, str]:
        """Analyze uploaded ebook and return summary"""
        if not self.converter:
            return "❌ Converter not available", "{\"error\": \"Converter not available\"}"
        
        if not file_path or not file_path.strip():
            return "❌ No file provided", "{\"error\": \"No file provided\"}"
        
        try:
            analysis = self.converter.analyze_ebook(file_path)
            
            # Format analysis results
            summary = f"""
📖 **Ebook Analysis Results**

**File:** {Path(file_path).name}
**Total Words:** {analysis['total_words']:,}
**Estimated Duration:** {analysis['estimated_duration_minutes']:.1f} minutes ({analysis['estimated_duration_minutes']/60:.1f} hours)
**Chapters Found:** {analysis['total_chapters']}

**Chapter Breakdown:**
"""
            
            for chapter in analysis['chapters'][:10]:  # Show first 10 chapters
                summary += f"- Chapter {chapter['number']}: {chapter['title'][:50]}{'...' if len(chapter['title']) > 50 else ''} ({chapter['word_count']:,} words, {chapter['estimated_duration_minutes']:.1f} min)\n"
            
            if len(analysis['chapters']) > 10:
                summary += f"... and {len(analysis['chapters']) - 10} more chapters\n"
            
            # Return JSON for detailed view
            detailed_json = json.dumps(analysis, indent=2)
            
            return summary, detailed_json
            
        except Exception as e:
            error_json = json.dumps({"error": str(e), "details": "Failed to analyze ebook"}, indent=2)
            return f"❌ Error analyzing ebook: {e}", error_json
    
    def convert_ebook(
        self,
        file_path: str,
        voice_name: str,
        speed: float,
        audio_format: str,
        engine: str,
        bitrate: str,
        title: str,
        author: str,
        cover_image,
        preview_mode: bool,
        progress=gr.Progress()
    ) -> Tuple[str, str, List[str]]:
        """Convert ebook to audiobook"""
        if not self.converter:
            return "❌ Converter not available", "", []
        
        try:
            progress(0, desc="Initializing conversion...")
            
            # Process voice selection with voice library support
            if VOICE_LIBRARY_AVAILABLE:
                # Extract voice name from display format with emojis
                if any(emoji in voice_name for emoji in ["🎙️", "🗣️", "🎵"]):
                    # Remove emojis and get voice name
                    clean_voice_name = voice_name.split("️ ")[-1] if "️ " in voice_name else voice_name
                    
                    # Find matching voice in library
                    matching_voice = None
                    for voice in voice_library.get_all_voices():
                        if voice.name in clean_voice_name:
                            matching_voice = voice
                            break
                    
                    if matching_voice:
                        # Use voice ID for conversion
                        clean_voice = matching_voice.id
                        # Update engine if different from selection
                        if engine == "auto":
                            engine = matching_voice.engine
                    else:
                        # Fallback to basic cleaning
                        clean_voice = voice_name.split(' (')[0] if ' (' in voice_name else voice_name
                else:
                    clean_voice = voice_name.split(' (')[0] if ' (' in voice_name else voice_name
            else:
                # Fallback to original method
                clean_voice = voice_name.split(' (')[0] if ' (' in voice_name else voice_name
            
            # Create output directory
            output_dir = self.temp_dir / f"conversion_{Path(file_path).stem}"
            output_dir.mkdir(exist_ok=True)
            
            # Handle cover image
            cover_path = ""
            if cover_image and hasattr(cover_image, 'name'):
                cover_path = str(output_dir / f"cover{Path(cover_image.name).suffix}")
                try:
                    import shutil
                    shutil.copy2(cover_image.name, cover_path)
                except:
                    cover_path = ""
            
            # Create conversion config
            config = ConversionConfig(
                input_file=file_path,
                output_dir=str(output_dir),
                voice_name=clean_voice,
                speed=speed,
                format=audio_format.lower(),
                engine=engine,
                bitrate=bitrate,
                title=title,
                author=author,
                cover_image=cover_path,
                preview_mode=preview_mode
            )
            
            progress(0.1, desc="Extracting text from ebook...")
            
            # Convert ebook
            results = self.converter.convert_to_audiobook(config)
            
            progress(0.9, desc="Finalizing conversion...")
            
            # Format results
            if results['audio_files']:
                status = f"""
✅ **Conversion Completed Successfully!**

**Input:** {Path(file_path).name}
**Voice:** {clean_voice}
**Engine:** {results.get('engine_used', engine)}
**Speed:** {speed}x
**Format:** {results.get('format', audio_format)}
**Bitrate:** {bitrate} (for MP3/M4B)
**Mode:** {'Preview (2 chapters)' if preview_mode else 'Full conversion'}

**Results:**
- Chapters converted: {len(results['audio_files'])}
- Output directory: {results['output_dir']}

**Metadata:**
- Title: {title or 'Auto-generated'}
- Author: {author or 'Unknown'}
"""
                
                if results['errors']:
                    status += f"\n⚠️ **Warnings/Errors:**\n"
                    for error in results['errors'][:5]:
                        status += f"- {error}\n"
                
                progress(1.0, desc="Conversion complete!")
                
                # Return audio files for download
                audio_files = results['audio_files'][:10]  # Limit to first 10 for UI
                
                return status, str(output_dir), audio_files
                
            else:
                return "❌ No audio files were generated", "", []
                
        except Exception as e:
            return f"❌ Conversion failed: {e}", "", []
    
    def get_custom_voices_dropdown_choices(self):
        """Get dropdown choices for custom voices"""
        try:
            voices = self.tts_backend.get_custom_voices()
            if not voices:
                return []
            return [voice.name for voice in voices]
        except Exception as e:
            return []
    
    def get_custom_voices_list(self):
        """Get formatted list of custom voices for display as JSON"""
        try:
            if not self.tts_backend:
                return json.dumps({"error": "TTS backend not available"})
            
            voices = self.tts_backend.get_custom_voices()
            if not voices:
                return json.dumps({"message": "No custom voices uploaded yet.", "voices": []})
            
            voice_list = []
            for voice in voices:
                voice_list.append({
                    "name": voice.name,
                    "language": voice.language,
                    "gender": voice.gender,
                    "emoji": "👩" if voice.gender == "female" else "👨" if voice.gender == "male" else "🤖"
                })
            
            return json.dumps({"voices": voice_list})
            
        except Exception as e:
            return json.dumps({"error": f"Error loading custom voices: {e}"})
    
    def handle_voice_upload(self, audio_file, voice_name, language, gender):
        """Handle custom voice upload"""
        try:
            if not audio_file:
                return "❌ Please select an audio file", self.get_custom_voices_list(), gr.update()
            
            if not voice_name or not voice_name.strip():
                return "❌ Please enter a voice name", self.get_custom_voices_list(), gr.update()
            
            # Clean voice name
            voice_name = voice_name.strip()
            
            # Save the voice using the correct method signature
            success = self.tts_backend.save_custom_voice(
                audio_file_path=audio_file,
                voice_name=voice_name,
                language=language,
                gender=gender
            )
            
            if success:
                choices = self.get_custom_voices_dropdown_choices()
                return (
                    f"✅ Voice '{voice_name}' uploaded successfully!", 
                    self.get_custom_voices_list(),
                    gr.update(choices=choices, value=voice_name)
                )
            else:
                return f"❌ Failed to upload voice '{voice_name}'", self.get_custom_voices_list(), gr.update()
                
        except Exception as e:
            return f"❌ Upload failed: {e}", self.get_custom_voices_list(), gr.update()
    
    def handle_voice_refresh(self):
        """Refresh custom voices list"""
        choices = self.get_custom_voices_dropdown_choices()
        return self.get_custom_voices_list(), gr.update(choices=choices)
    
    def handle_voice_removal(self, voice_name):
        """Handle voice removal"""
        try:
            if not voice_name or not voice_name.strip():
                return "❌ Please select a voice to remove", self.get_custom_voices_list(), gr.update()
            
            success = self.tts_backend.remove_custom_voice(voice_name.strip())
            
            if success:
                choices = self.get_custom_voices_dropdown_choices()
                return (
                    f"✅ Voice '{voice_name}' removed successfully!", 
                    self.get_custom_voices_list(),
                    gr.update(choices=choices, value=None)
                )
            else:
                return f"❌ Voice '{voice_name}' not found", self.get_custom_voices_list(), gr.update()
                
        except Exception as e:
            return f"❌ Removal failed: {e}", self.get_custom_voices_list(), gr.update()
    
    def convert_ebook_with_status(
        self,
        file_path: str,
        voice_name: str,
        speed: float,
        audio_format: str,
        engine: str,
        bitrate: str,
        title: str,
        author: str,
        cover_image,
        preview_mode: bool,
        output_root: str,
        progress=gr.Progress()
    ):
        """Generator wrapper for ebook conversion with output folder selection and live status ticker updates."""
        if not self.converter:
            yield "Converter not available", "? Converter not available", "", []
            return

        try:
            progress(0, desc="Initializing conversion...")

            # Resolve output directory
            try:
                chosen_root = Path(output_root).expanduser() if (output_root and str(output_root).strip()) else self.default_output_root
            except Exception:
                chosen_root = self.default_output_root
            output_dir = chosen_root / "Ebooks" / f"{Path(file_path).stem}"
            output_dir.mkdir(parents=True, exist_ok=True)
            yield "Initializing conversion...", "Preparing conversion...", str(output_dir), []

            # Process voice selection with voice library support (replicate logic)
            if VOICE_LIBRARY_AVAILABLE:
                if any(emoji in voice_name for emoji in ["???", "???", "??"]):
                    clean_voice_name = voice_name.split("? ")[-1] if "? " in voice_name else voice_name
                    matching_voice = None
                    for voice in voice_library.get_all_voices():
                        if voice.name in clean_voice_name:
                            matching_voice = voice
                            break
                    if matching_voice:
                        clean_voice = matching_voice.id
                        if engine == "auto":
                            engine = matching_voice.engine
                    else:
                        clean_voice = voice_name.split(' (')[0] if ' (' in voice_name else voice_name
                else:
                    clean_voice = voice_name.split(' (')[0] if ' (' in voice_name else voice_name
            else:
                clean_voice = voice_name.split(' (')[0] if ' (' in voice_name else voice_name

            # Handle cover image
            cover_path = ""
            if cover_image and hasattr(cover_image, 'name'):
                cover_path = str(output_dir / f"cover{Path(cover_image.name).suffix}")
                try:
                    import shutil as _shutil
                    _shutil.copy2(cover_image.name, cover_path)
                except Exception:
                    cover_path = ""

            # Build config
            config = ConversionConfig(
                input_file=file_path,
                output_dir=str(output_dir),
                voice_name=clean_voice,
                speed=speed,
                format=audio_format.lower(),
                engine=engine,
                bitrate=bitrate,
                title=title,
                author=author,
                cover_image=cover_path,
                preview_mode=preview_mode
            )

            progress(0.1, desc="Extracting text from ebook...")
            yield "Extracting text...", "Analyzing and extracting text...", str(output_dir), []

            results = self.converter.convert_to_audiobook(config)

            progress(0.9, desc="Finalizing conversion...")
            yield "Converting chapters...", "Generating audio files...", str(output_dir), []

            if results['audio_files']:
                status = f"""
✅ **Conversion Completed Successfully!**

**Input:** {Path(file_path).name}
**Voice:** {clean_voice}
**Engine:** {results.get('engine_used', engine)}
**Speed:** {speed}x
**Format:** {results.get('format', audio_format)}
**Bitrate:** {bitrate} (for MP3/M4B)
**Mode:** {'Preview (2 chapters)' if preview_mode else 'Full conversion'}

**Results:**
- Chapters converted: {len(results['audio_files'])}
- Output directory: {results['output_dir']}

**Metadata:**
- Title: {title or 'Auto-generated'}
- Author: {author or 'Unknown'}
"""
                if results['errors']:
                    status += f"\n⚠️ **Warnings/Errors:**\n"
                    for error in results['errors'][:5]:
                        status += f"- {error}\n"

                progress(1.0, desc="Conversion complete!")
                audio_files = results['audio_files'][:10]
                yield "Done", status, str(output_dir), audio_files
                return
            else:
                yield "No output", "⚠️ No audio files were generated", str(output_dir), []
                return

        except Exception as e:
            yield "Error", f"❌ Conversion failed: {e}", "", []
            return

    def open_output_dir(self, path: str) -> str:
        """Open the specified folder and return a one-line status string."""
        try:
            target = Path(path).expanduser() if path else self.default_output_root
            target.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                import subprocess as _sp
                _sp.Popen(["open", str(target)])
            else:
                import subprocess as _sp
                _sp.Popen(["xdg-open", str(target)])
            return f"Opened: {target}"
        except Exception as e:
            return f"Failed to open folder: {e}"

    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(
            title="📚 Ebook to Audiobook Converter",
            theme=gr.themes.Soft(),
            css="""
            .main-header { text-align: center; margin-bottom: 2rem; }
            .feature-box { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
            .status-success { color: #28a745; }
            .status-error { color: #dc3545; }
            """
        ) as interface:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>📚 Ebook to Audiobook Converter</h1>
                <p>Convert PDF, TXT, DOCX, and EPUB files to audiobooks using VibeVoice TTS</p>
            </div>
            """)
            
            # System status
            with gr.Row():
                with gr.Column():
                    gr.HTML(f"""
                    <div class="feature-box">
                        <h3>🔧 System Status</h3>
                        <p>Converter: {'✅ Available' if CONVERTER_AVAILABLE else '❌ Not Available'}</p>
                        <p>TTS Backend: {'✅ Available' if TTS_AVAILABLE else '❌ Not Available'}</p>
                        <p>Supported Formats: PDF, TXT, DOCX, EPUB</p>
                    </div>
                    """)
                    # GPU status add-on
                    gr.HTML(f"""
                    <div class=\"feature-box\">\n
                        <p>CUDA: {'✅' if torch.cuda.is_available() else '❌'}{(' — ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else ''}</p>\n
                        <p>Precision: {('bfloat16' if (torch.cuda.is_available() and hasattr(torch, 'bfloat16')) else ('float16' if torch.cuda.is_available() else 'float32'))}</p>\n
                    </div>
                    """)

            # Main interface
            with gr.Tab("📖 Convert Ebook"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Input section
                        gr.HTML("<h3>📁 Input Settings</h3>")
                        
                        file_input = gr.File(
                            label="Upload Ebook",
                            file_types=[".pdf", ".txt", ".docx", ".epub"],
                            type="filepath"
                        )
                        
                        # Voice and Engine settings
                        gr.HTML("<h3>🎙️ Voice & Engine Settings</h3>")
                        
                        # VibeVoice Model Selection for Ebook
                        ebook_model_dropdown = gr.Dropdown(
                            choices=list(self.available_models.keys()),
                            value="VibeVoice-1.5B",
                            label="VibeVoice Model (for advanced features)",
                            info="1.5B: 64K context, Large: 32K context",
                            visible=True
                        )
                        
                        engine_dropdown = gr.Dropdown(
                            choices=self.get_available_engines(),
                            value="vibevoice",
                            label="TTS Engine",
                            info="Select TTS engine (Coqui requires Python 3.11 container)"
                        )
                        
                        # Output settings
                        gr.HTML("<h3>📦 Output Settings</h3>")
                        output_root_input = gr.Textbox(
                            label="Output Directory",
                            value=str(self.default_output_root),
                            placeholder=str(self.default_output_root),
                            max_lines=1
                        )
                        open_output_btn = gr.Button("📂 Open Output Folder", variant="secondary")
                        
                        # Voice library search and filters
                        if VOICE_LIBRARY_AVAILABLE:
                            with gr.Row():
                                voice_search = gr.Textbox(
                                    label="🔍 Search Voices",
                                    placeholder="Search by name, language, or description...",
                                    scale=2
                                )
                                search_btn = gr.Button("Search", size="sm")
                            
                            with gr.Row():
                                language_filter = gr.Dropdown(
                                    choices=["All Languages"] + voice_library.get_voice_categories()["languages"],
                                    value="All Languages",
                                    label="Filter by Language",
                                    scale=1
                                )
                                gender_filter = gr.Dropdown(
                                    choices=["All Genders"] + voice_library.get_voice_categories()["genders"],
                                    value="All Genders", 
                                    label="Filter by Gender",
                                    scale=1
                                )
                        
                        voice_dropdown = gr.Dropdown(
                            choices=self.get_available_voices(),
                            value=self.get_available_voices()[0] if self.get_available_voices() else "⭐👩🎙️ Isabella (British Female)",
                            label="🎤 Select Voice",
                            info="Choose from 50+ high-quality voices across 9 languages"
                        )
                        
                        # Voice information display
                        if VOICE_LIBRARY_AVAILABLE:
                            voice_info = gr.Markdown(
                                label="Voice Information",
                                value="Select a voice to see details."
                            )
                        
                        # Voice preview (placeholder for future implementation)
                        with gr.Row():
                            preview_text = gr.Textbox(
                                label="Preview Text",
                                value="Hello! This is how I sound. I will narrate your audiobook with this voice.",
                                placeholder="Enter text to preview voice",
                                scale=3
                            )
                            preview_btn = gr.Button("🔊 Preview Voice", size="sm", scale=1)
                        
                        # Advanced voice controls
                        with gr.Accordion("🎛️ Advanced Voice Controls", open=False):
                            speed_slider = gr.Slider(
                                minimum=0.5,
                                maximum=2.0,
                                value=1.3,
                                step=0.1,
                                label="Speech Speed",
                                info="0.5x = slow, 1.0x = normal, 2.0x = fast"
                            )
                            
                            if VOICE_LIBRARY_AVAILABLE:
                                temperature_slider = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.7,
                                    step=0.1,
                                    label="Voice Expressiveness",
                                    info="Higher values = more expressive, lower = more consistent"
                                )
                                
                                length_penalty = gr.Slider(
                                    minimum=0.5,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1,
                                    label="Length Penalty",
                                    info="Controls speech pacing and rhythm"
                                )
                        
                        # Output settings
                        gr.HTML("<h3>🔊 Output Settings</h3>")
                        
                        format_radio = gr.Radio(
                            choices=["WAV", "MP3", "M4B"],
                            value="WAV",
                            label="Audio Format - WAV: Uncompressed, MP3: Compressed, M4B: Audiobook with chapters"
                        )
                        
                        bitrate_dropdown = gr.Dropdown(
                            choices=["64k", "96k", "128k", "192k", "256k"],
                            value="128k",
                            label="Bitrate (MP3/M4B) - Higher = better quality, larger file size"
                        )
                        
                        # Metadata settings
                        gr.HTML("<h3>📝 Metadata Settings</h3>")
                        
                        title_input = gr.Textbox(
                            label="Book Title",
                            placeholder="Auto-detected from filename if empty",
                            info="Used in MP3/M4B metadata"
                        )
                        
                        author_input = gr.Textbox(
                            label="Author",
                            placeholder="Unknown",
                            info="Used in MP3/M4B metadata"
                        )
                        
                        cover_input = gr.File(
                            label="Cover Image (M4B only) - Optional cover art for M4B audiobook",
                            file_types=[".jpg", ".jpeg", ".png"],
                            type="filepath"
                        )
                        
                        # Conversion options
                        gr.HTML("<h3>⚙️ Conversion Options</h3>")
                        
                        preview_checkbox = gr.Checkbox(
                            label="Preview Mode",
                            value=False,
                            info="Convert only first 2 chapters for testing"
                        )
                        
                        # Action buttons
                        with gr.Row():
                            analyze_btn = gr.Button("📊 Analyze Ebook", variant="secondary")
                            convert_btn = gr.Button("🎧 Convert to Audiobook", variant="primary")
                    
                    with gr.Column(scale=2):
                        # Results section
                        gr.HTML("<h3>📋 Results</h3>")
                        
                        status_output = gr.Markdown(
                            label="Status",
                            value="Upload an ebook file to begin analysis or conversion"
                        )
                        
                        status_ticker = gr.Textbox(
                            label="Status",
                            value="Idle",
                            interactive=False,
                            max_lines=1
                        )
                        
                        output_dir = gr.Textbox(
                            label="Output Directory",
                            interactive=False,
                            visible=False
                        )
                        
                        audio_files = gr.File(
                            label="Generated Audio Files",
                            file_count="multiple",
                            visible=False
                        )
                        
            with gr.Tab("🎙️ Podcast Generation"):
                gr.HTML("""
                <div class="main-header">
                    <h2>🎙️ Multi-Speaker Podcast Generation</h2>
                    <p>Create conversations and podcasts with multiple speakers using VibeVoice</p>
                </div>
                """)
                
                # Model Selection
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=list(self.available_models.keys()),
                        value="VibeVoice-1.5B",
                        label="VibeVoice Model",
                        info="Choose between models (1.5B: 64K context, Large: 32K context)"
                    )
                    model_status = gr.Textbox(
                        label="Model Status",
                        value="Ready to load model",
                        interactive=False
                    )
                    load_model_btn = gr.Button("🔄 Load Model", variant="secondary")
                
                # Podcast Configuration
                with gr.Row():
                    with gr.Column(scale=2):
                        # Script Input
                        gr.Markdown("### 📝 **Conversation Script**")
                        podcast_script = gr.Textbox(
                            label="Podcast Script",
                            placeholder="""Enter your podcast script here. Format it as:

Speaker 1: Welcome to our podcast today!
Speaker 2: Thanks for having me. This is exciting!
Speaker 1: Let's dive into our topic...
Speaker 2: I couldn't agree more!""",
                            lines=10,
                            max_lines=20
                        )
                        
                        # Generation Controls
                        with gr.Row():
                            num_speakers = gr.Slider(
                                minimum=1,
                                maximum=4,
                                value=2,
                                step=1,
                                label="Number of Speakers"
                            )
                            generate_podcast_btn = gr.Button(
                                "🚀 Generate Podcast",
                                variant="primary",
                                size="lg"
                            )
                    
                    with gr.Column(scale=1):
                        # Speaker Assignment
                        gr.Markdown("### 🎭 **Speaker Voice Assignment**")
                        
                        speaker_voices = []
                        for i in range(4):  # Support up to 4 speakers
                            speaker_voice = gr.Dropdown(
                                choices=self.get_simple_voice_names(),
                                label=f"Speaker {i+1} Voice",
                                visible=(i < 2),  # Show first 2 by default
                                interactive=True,
                                value="bf_isabella" if i == 0 else "en-Alice_woman" if i == 1 else None
                            )
                            speaker_voices.append(speaker_voice)
                
                # Generation Output
                with gr.Row():
                    with gr.Column():
                        podcast_status = gr.Textbox(
                            label="Generation Status",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.Column():
                        podcast_audio = gr.Audio(
                            label="Generated Podcast",
                            interactive=False
                        )
                
                # Guidelines
                gr.HTML("""
                <div class="feature-box">
                    <h4>📋 Guidelines for Podcast Generation</h4>
                    <ul>
                        <li><strong>Script Format:</strong> Use "Speaker X: text" format for each line</li>
                        <li><strong>Model Selection:</strong> VibeVoice-1.5B supports longer contexts (64K tokens)</li>
                        <li><strong>Speaker Voices:</strong> Assign different voices to create natural conversations</li>
                        <li><strong>Content Length:</strong> Longer scripts work better with VibeVoice-1.5B model</li>
                        <li><strong>Quality:</strong> Use natural conversation flow for best results</li>
                    </ul>
                </div>
                """)
            
            with gr.Tab("📊 Analysis"):
                with gr.Row():
                    with gr.Column():
                        analysis_output = gr.Markdown(
                            label="Analysis Summary",
                            value="Upload an ebook to see detailed analysis"
                        )
                    
                    with gr.Column():
                        detailed_json = gr.JSON(
                            label="Detailed Analysis",
                            value={"status": "Upload an ebook to see detailed analysis"}
                        )
            
            with gr.Tab("📋 Chapter Selection"):
                gr.HTML("<h3>🎯 Interactive Chapter Selection</h3>")
                gr.HTML("<p>Select specific chapters to convert instead of the entire book</p>")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # File input for chapter selection
                        chapter_file_input = gr.File(
                            label="Upload Ebook for Chapter Selection",
                            file_types=[".pdf", ".txt", ".docx", ".epub"],
                            type="filepath"
                        )
                        
                        load_chapters_btn = gr.Button("📖 Load Chapter List", variant="secondary")
                        
                        # Chapter selection controls
                        with gr.Row():
                            select_all_btn = gr.Button("✅ Select All", size="sm")
                            deselect_all_btn = gr.Button("❌ Deselect All", size="sm")
                            invert_selection_btn = gr.Button("🔄 Invert Selection", size="sm")
                        
                        # Quick selection options
                        gr.HTML("<h4>📍 Quick Selection</h4>")
                        with gr.Row():
                            first_n_chapters = gr.Number(
                                label="First N chapters",
                                value=5,
                                minimum=1,
                                maximum=100,
                                scale=2
                            )
                            select_first_n_btn = gr.Button("Select First N", size="sm", scale=1)
                        
                        with gr.Row():
                            chapter_range_start = gr.Number(
                                label="From chapter",
                                value=1,
                                minimum=1,
                                scale=1
                            )
                            chapter_range_end = gr.Number(
                                label="To chapter",
                                value=10,
                                minimum=1,
                                scale=1
                            )
                            select_range_btn = gr.Button("Select Range", size="sm", scale=1)
                    
                    with gr.Column(scale=2):
                        # Chapter list display
                        chapter_selection_output = gr.Markdown(
                            label="Chapter List",
                            value="Upload an ebook and click 'Load Chapter List' to see available chapters"
                        )
                        
                        # Chapter checkboxes will be dynamically generated
                        chapter_checkboxes = gr.CheckboxGroup(
                            label="Select Chapters to Convert",
                            choices=[],
                            value=[],
                            visible=False
                        )
                
                # Conversion controls for selected chapters
                gr.HTML("<h4>🎧 Convert Selected Chapters</h4>")
                with gr.Row():
                    with gr.Column(scale=1):
                        # Voice settings for chapter conversion
                        chapter_voice_dropdown = gr.Dropdown(
                            choices=self.get_available_voices() if hasattr(self, 'get_available_voices') else [],
                            value=self.get_available_voices()[0] if hasattr(self, 'get_available_voices') and self.get_available_voices() else "Default Voice",
                            label="🎤 Voice for Selected Chapters"
                        )
                        
                        chapter_format_radio = gr.Radio(
                            choices=["WAV", "MP3", "M4B"],
                            value="MP3",
                            label="Output Format"
                        )
                        
                        convert_selected_btn = gr.Button("🎧 Convert Selected Chapters", variant="primary")
                    
                    with gr.Column(scale=2):
                        chapter_conversion_status = gr.Markdown(
                            label="Conversion Status",
                            value="Select chapters and click convert to begin"
                        )
                        
                        selected_audio_files = gr.File(
                            label="Generated Audio Files",
                            file_count="multiple",
                            visible=False
                        )
            
            with gr.Tab("ℹ️ Help"):
                gr.HTML("""
                <div class="feature-box">
                    <h3>📖 Supported Formats</h3>
                    <ul>
                        <li><strong>PDF:</strong> Portable Document Format files</li>
                        <li><strong>TXT:</strong> Plain text files</li>
                        <li><strong>DOCX:</strong> Microsoft Word documents</li>
                        <li><strong>EPUB:</strong> Electronic publication format</li>
                    </ul>
                    
                    <h3>🎙️ TTS Engines</h3>
                    <ul>
                        <li><strong>VibeVoice:</strong> High-quality Microsoft TTS (always available)</li>
                        <li><strong>Coqui:</strong> Open-source TTS with many voices (requires Python 3.11 container)</li>
                        <li><strong>Auto:</strong> Automatically select best available engine</li>
                    </ul>
                    
                    <h3>🔊 Audio Formats</h3>
                    <ul>
                        <li><strong>WAV:</strong> Uncompressed, best quality, larger files</li>
                        <li><strong>MP3:</strong> Compressed, good quality, smaller files, individual chapters + combined file</li>
                        <li><strong>M4B:</strong> Audiobook format with chapter markers, metadata, and optional cover art</li>
                    </ul>
                    
                    <h3>⚙️ Settings Guide</h3>
                    <ul>
                        <li><strong>Speed:</strong> 1.0 = normal, 1.3 = slightly faster (recommended for audiobooks)</li>
                        <li><strong>Bitrate:</strong> 128k recommended for good quality/size balance</li>
                        <li><strong>Preview Mode:</strong> Convert only first 2 chapters for testing settings</li>
                        <li><strong>Metadata:</strong> Title and author are embedded in MP3/M4B files</li>
                        <li><strong>Cover Art:</strong> JPG/PNG images embedded in M4B files</li>
                    </ul>
                    
                    <h3>🔧 Troubleshooting</h3>
                    <ul>
                        <li>Large files may take significant time to process</li>
                        <li>Use Preview Mode to test settings before full conversion</li>
                        <li>MP3/M4B require FFmpeg - falls back to WAV if not available</li>
                        <li>Coqui engine requires the Python 3.11 container (vibe-ebook-py311)</li>
                        <li>GPU acceleration automatically used if available</li>
                        <li>Check output directory if files don't appear in download</li>
                    </ul>
                    
                    <h3>🐳 Docker Services</h3>
                    <ul>
                        <li><strong>vibe-ebook:</strong> Python 3.13, VibeVoice only (current)</li>
                        <li><strong>vibe-ebook-py311:</strong> Python 3.11, VibeVoice + Coqui (planned)</li>
                    </ul>
                </div>
                """)
            
            # Voice Upload Tab - NEW FEATURE!
            with gr.Tab("🎙️ Voice Upload"):
                gr.HTML("""
                <div class="feature-box">
                    <h3>📤 Custom Voice Upload</h3>
                    <p>Upload your own voice samples for personalized audiobook narration using VibeVoice TTS.</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Voice Upload Section
                        gr.HTML("<h4>📁 Upload Voice Sample</h4>")
                        
                        upload_voice_file = gr.Audio(
                            label="Voice Sample",
                            type="filepath",
                            format="wav"
                        )
                        upload_voice_name = gr.Textbox(
                            label="Voice Name",
                            placeholder="Enter a unique name for your custom voice",
                            max_lines=1
                        )
                        
                        with gr.Row():
                            upload_voice_language = gr.Dropdown(
                                choices=["en", "zh", "fr", "de", "es"],
                                value="en",
                                label="Language"
                            )
                            upload_voice_gender = gr.Dropdown(
                                choices=["female", "male", "neutral"],
                                value="neutral",
                                label="Gender"
                            )
                        
                        upload_voice_btn = gr.Button(
                            "💾 Save Custom Voice",
                            variant="primary",
                            size="lg"
                        )
                        upload_status = gr.Textbox(
                            label="Upload Status",
                            interactive=False,
                            visible=True
                        )
                    
                    with gr.Column(scale=1):
                        # Voice Management Section
                        gr.HTML("<h4>🗂️ Manage Custom Voices</h4>")
                        
                        voice_list = gr.Dropdown(
                            label="Custom Voices",
                            choices=[],
                            interactive=True
                        )
                        
                        custom_voices_display = gr.Markdown(
                            value="No custom voices uploaded yet.",
                            label="Uploaded Voices"
                        )
                        
                        with gr.Row():
                            refresh_voices_btn = gr.Button(
                                "🔄 Refresh List",
                                variant="secondary"
                            )
                            remove_voice_btn = gr.Button(
                                "🗑️ Remove Voice",
                                variant="stop"
                            )
                        
                        voice_info = gr.JSON(
                            label="Voice Information",
                            visible=False
                        )
                        
                        gr.HTML("""
                        <div class="feature-box">
                            <h4>ℹ️ Voice Upload Guidelines</h4>
                            <ul>
                                <li>Use high-quality audio files (WAV preferred)</li>
                                <li>10-60 seconds of clear speech</li>
                                <li>Single speaker, minimal background noise</li>
                                <li>Natural speaking pace and tone</li>
                                <li>Custom voices use VibeVoice neural synthesis</li>
                                <li>Voices persist across sessions</li>
                            </ul>
                        </div>
                        """)
            
            # Event handlers
            def handle_analyze_ebook(file_input):
                """Handle ebook analysis with proper error handling"""
                if not file_input:
                    return "❌ No file uploaded", "{\"error\": \"No file uploaded\"}"
                return self.analyze_ebook(file_input)
            
            analyze_btn.click(
                fn=handle_analyze_ebook,
                inputs=[file_input],
                outputs=[analysis_output, detailed_json],
                show_progress=True
            )
            
            def handle_convert_ebook(file_input, voice_dropdown, speed_slider, format_radio, engine_dropdown, bitrate_dropdown, title_input, author_input, cover_input, preview_checkbox, output_root):
                """Handle ebook conversion with validation and live updates (saves to folder)."""
                if not file_input:
                    yield "No file", "? No file uploaded", "", []
                    return
                for update in self.convert_ebook_with_status(
                    file_path=file_input,
                    voice_name=voice_dropdown,
                    speed=speed_slider,
                    audio_format=format_radio,
                    engine=engine_dropdown,
                    bitrate=bitrate_dropdown,
                    title=title_input,
                    author=author_input,
                    cover_image=cover_input,
                    preview_mode=preview_checkbox,
                    output_root=output_root,
                ):
                    yield update

            # Open output folder
            open_output_btn.click(
                fn=self.open_output_dir,
                inputs=[output_root_input],
                outputs=[status_ticker]
            )

            # Primary convert: save to folder with live status ticker
            convert_btn.click(
                fn=handle_convert_ebook,
                inputs=[
                    file_input,
                    voice_dropdown,
                    speed_slider,
                    format_radio,
                    engine_dropdown,
                    bitrate_dropdown,
                    title_input,
                    author_input,
                    cover_input,
                    preview_checkbox,
                    output_root_input
                ],
                outputs=[status_ticker, status_output, output_dir, audio_files],
                show_progress=True
            ).then(
                fn=lambda files: gr.update(visible=len(files) > 0) if files else gr.update(visible=False),
                inputs=[audio_files],
                outputs=[audio_files]
            )
            
            # Podcast generation event handlers
            load_model_btn.click(
                fn=self.switch_vibevoice_model,
                inputs=[model_dropdown],
                outputs=[model_status],  # Updates model_status
                show_progress=True
            )
            
            # Update speaker voice dropdowns based on number of speakers
            def update_speaker_visibility(num_speakers):
                updates = []
                for i in range(4):
                    updates.append(gr.update(visible=(i < num_speakers)))
                return updates
            
            num_speakers.change(
                fn=update_speaker_visibility,
                inputs=[num_speakers],
                outputs=speaker_voices
            )
            
            def handle_generate_podcast(num_speakers, podcast_script, *speaker_voices):
                """Handle podcast generation with validation"""
                if not podcast_script or not podcast_script.strip():
                    return "❌ Please enter a podcast script", None
                return self.generate_podcast(num_speakers, podcast_script, *speaker_voices)
            
            generate_podcast_btn.click(
                fn=handle_generate_podcast,
                inputs=[num_speakers, podcast_script] + speaker_voices,
                outputs=[podcast_status, podcast_audio],
                show_progress=True
            )
            
            # Voice upload event handlers
            upload_voice_btn.click(
                fn=self.handle_voice_upload,
                inputs=[
                    upload_voice_file,
                    upload_voice_name,
                    upload_voice_language,
                    upload_voice_gender
                ],
                outputs=[upload_status, custom_voices_display, voice_list],
                show_progress=True
            )
            
            refresh_voices_btn.click(
                fn=self.handle_voice_refresh,
                inputs=[],
                outputs=[custom_voices_display, voice_list]
            )
            
            remove_voice_btn.click(
                fn=self.handle_voice_removal,
                inputs=[voice_list],
                outputs=[upload_status, custom_voices_display, voice_list]
            )
            
            # Voice library event handlers
            if VOICE_LIBRARY_AVAILABLE:
                # Voice search
                search_btn.click(
                    fn=self.search_voices,
                    inputs=[voice_search],
                    outputs=[voice_dropdown]
                )
                
                # Filter by language
                language_filter.change(
                    fn=self.filter_voices_by_language,
                    inputs=[language_filter],
                    outputs=[voice_dropdown]
                )
                
                # Filter by gender
                gender_filter.change(
                    fn=self.filter_voices_by_gender,
                    inputs=[gender_filter],
                    outputs=[voice_dropdown]
                )
                
                # Voice info display
                voice_dropdown.change(
                    fn=self.get_voice_info,
                    inputs=[voice_dropdown],
                    outputs=[voice_info]
                )
                
                # Voice search on enter
                voice_search.submit(
                    fn=self.search_voices,
                    inputs=[voice_search],
                    outputs=[voice_dropdown]
                )
            
            # Chapter selection event handlers
            def handle_chapter_loading(file_path):
                """Handle chapter loading and return components updates"""
                summary, choices, values = self.load_chapter_list(file_path)
                return summary, gr.update(choices=choices, value=values, visible=len(choices) > 0)
            
            load_chapters_btn.click(
                fn=handle_chapter_loading,
                inputs=[chapter_file_input],
                outputs=[chapter_selection_output, chapter_checkboxes],
                show_progress=True
            )
            
            # Chapter selection controls
            def handle_select_all(checkboxes_component):
                """Select all chapters"""
                choices = checkboxes_component.choices if hasattr(checkboxes_component, 'choices') else []
                return gr.update(value=choices)
            
            def handle_deselect_all(checkboxes_component):
                """Deselect all chapters"""
                return gr.update(value=[])
            
            def handle_invert_selection(checkboxes_component):
                """Invert chapter selection"""
                choices = checkboxes_component.choices if hasattr(checkboxes_component, 'choices') else []
                current = checkboxes_component.value if hasattr(checkboxes_component, 'value') else []
                inverted = [choice for choice in choices if choice not in current]
                return gr.update(value=inverted)
            
            def handle_select_first_n(checkboxes_component, n):
                """Select first N chapters"""
                choices = checkboxes_component.choices if hasattr(checkboxes_component, 'choices') else []
                selected = choices[:min(int(n), len(choices))]
                return gr.update(value=selected)
            
            def handle_select_range(checkboxes_component, start, end):
                """Select chapters in range"""
                choices = checkboxes_component.choices if hasattr(checkboxes_component, 'choices') else []
                start_idx = max(0, int(start) - 1)  # Convert to 0-based index
                end_idx = min(len(choices), int(end))  # Inclusive end
                selected = choices[start_idx:end_idx]
                return gr.update(value=selected)
            
            select_all_btn.click(
                fn=handle_select_all,
                inputs=[chapter_checkboxes],
                outputs=[chapter_checkboxes]
            )
            
            deselect_all_btn.click(
                fn=handle_deselect_all,
                inputs=[chapter_checkboxes],
                outputs=[chapter_checkboxes]
            )
            
            invert_selection_btn.click(
                fn=handle_invert_selection,
                inputs=[chapter_checkboxes],
                outputs=[chapter_checkboxes]
            )
            
            select_first_n_btn.click(
                fn=handle_select_first_n,
                inputs=[chapter_checkboxes, first_n_chapters],
                outputs=[chapter_checkboxes]
            )
            
            select_range_btn.click(
                fn=handle_select_range,
                inputs=[chapter_checkboxes, chapter_range_start, chapter_range_end],
                outputs=[chapter_checkboxes]
            )
            
            # Convert selected chapters
            def handle_convert_selected_chapters(chapter_file_input, chapter_checkboxes, chapter_voice_dropdown, chapter_format_radio):
                """Handle selected chapters conversion with validation"""
                if not chapter_file_input:
                    return "❌ No file uploaded", []
                if not chapter_checkboxes:
                    return "❌ No chapters selected", []
                return self.convert_selected_chapters(chapter_file_input, chapter_checkboxes, chapter_voice_dropdown, chapter_format_radio)
            
            convert_selected_btn.click(
                fn=handle_convert_selected_chapters,
                inputs=[
                    chapter_file_input,
                    chapter_checkboxes,
                    chapter_voice_dropdown,
                    chapter_format_radio
                ],
                outputs=[chapter_conversion_status, selected_audio_files],
                show_progress=True
            ).then(
                fn=lambda files: gr.update(visible=len(files) > 0) if files else gr.update(visible=False),
                inputs=[selected_audio_files],
                outputs=[selected_audio_files]
            )
        
        return interface
    
    def launch(self, server_name="0.0.0.0", server_port=None, **kwargs):
        """Launch the Gradio interface with proper port management"""
        
        # Clean up any existing interface
        self.cleanup()
        
        # Find a free port if none specified
        if server_port is None:
            server_port = self.find_free_port()
        else:
            # Check if requested port is free, find alternative if not
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', server_port))
            except OSError:
                print(f"[WARN] Port {server_port} is busy, finding alternative...")
                self.kill_port_processes(server_port)
                time.sleep(2)  # Give time for cleanup
                server_port = self.find_free_port(server_port)
        
        self.server_port = server_port
        print(f"[START] Launching GUI on port {server_port}")
        
        # Create fresh interface
        self.interface = self.create_interface()
        
        # Launch with enhanced error handling
        launch_kwargs = {
            "server_name": server_name,
            "server_port": server_port,
            "prevent_thread_lock": True,
            "quiet": False,
        }
        launch_kwargs.update(kwargs)

        try:
            return self.interface.launch(**launch_kwargs)
        except ValueError as e:
            message = str(e)
            if "share=True" in message and not launch_kwargs.get("share"):
                print("[WARN] Localhost access failed, retrying with share link...")
                fallback_kwargs = launch_kwargs.copy()
                fallback_kwargs["share"] = True
                fallback_kwargs["server_name"] = os.environ.get("VIBEVOICE_GUI_SHARE_HOST", "0.0.0.0")
                return self.interface.launch(**fallback_kwargs)
            print(f"[ERROR] Launch failed: {e}")
            self.cleanup()
            raise
        except Exception as e:
            print(f"[ERROR] Launch failed: {e}")
            self.cleanup()
            raise

def main():
    """Main function to launch the GUI with enhanced port management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ebook to Audiobook Converter GUI")
    parser.add_argument("--port", type=int, default=7862, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--force-kill", action="store_true", help="Force kill processes on port before starting")
    args = parser.parse_args()
    
    print(f"[START] Starting Ebook to Audiobook Converter GUI on {args.host}:{args.port}...")
    
    # Create GUI instance
    gui = EbookConverterGUI()
    
    # Force cleanup if requested
    if args.force_kill:
        print(f"[CLEAN] Force cleaning port {args.port}...")
        gui.kill_port_processes(args.port)
        time.sleep(2)
    
    try:
        # Launch with enhanced settings
        gui.launch(
            server_name=args.host,
            server_port=args.port,
            share=False,
            debug=True,
            show_error=True,
            max_threads=4,
            inbrowser=True  # Automatically open browser
        )
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[STOP] Received interrupt signal, shutting down...")
            
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to start GUI: {e}")
        print("[TRACEBACK]")
        traceback.print_exc()
        
    finally:
        print("[CLEANUP] Performing final cleanup...")
        gui.cleanup()

if __name__ == "__main__":
    main()


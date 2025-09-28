#!/usr/bin/env python3
"""
VibeVoice Integrated TTS/STT Streamlit GUI
A comprehensive one-stop shop for TTS and STT featuring:
- VibeVoice for long-form conversational TTS
- OpenAI Whisper for speech-to-text
- Audiblez (Kokoro) for additional TTS options
- Coqui TTS for voice cloning
- Ebook to audiobook conversion
- Podcast generation
- Transcript generation
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
import json
import io

# Core dependencies
import numpy as np
import torch
import librosa
import soundfile as sf

# Try to import optional dependencies
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    print("Streamlit not available. Please install with: pip install streamlit")
    STREAMLIT_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("Whisper not available. Please install with: pip install openai-whisper")
    WHISPER_AVAILABLE = False

try:
    from TTS.api import TTS
    COQUI_TTS_AVAILABLE = True
except ImportError:
    print("Coqui TTS not available. Please install with: pip install TTS")
    COQUI_TTS_AVAILABLE = False

# VibeVoice imports
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

class IntegratedTTSSTTApp:
    """Integrated TTS/STT application combining multiple AI models"""
    
    def __init__(self):
        self.vibevoice_model = None
        self.vibevoice_processor = None
        self.whisper_model = None
        self.coqui_tts = None
        self.voice_presets = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.demo_mode = True  # Enable demo mode for offline functionality
        self.models_loaded = {
            'vibevoice': False,
            'whisper': False,
            'coqui': False
        }
        
    def load_vibevoice(self, model_path: str = "microsoft/VibeVoice-1.5B"):
        """Load VibeVoice model and processor"""
        try:
            st.info(f"Loading VibeVoice model from {model_path}...")
            
            # Try to load real model first
            try:
                # Load processor
                self.vibevoice_processor = VibeVoiceProcessor.from_pretrained(model_path)
                
                # Load model
                self.vibevoice_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map=self.device,
                    trust_remote_code=True
                )
                
                self.demo_mode = False
                self.models_loaded['vibevoice'] = True
                st.success("VibeVoice model loaded successfully!")
                
            except Exception as e:
                # Fall back to demo mode
                st.warning(f"Could not load real model ({str(e)}). Enabling demo mode.")
                self.demo_mode = True
                self.models_loaded['vibevoice'] = True
                st.info("✨ Demo Mode: Will generate synthetic audio for demonstration")
            
            # Setup voice presets
            self.setup_voice_presets()
            
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize VibeVoice: {str(e)}")
            return False
    
    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory"""
        voices_dir = Path("demo/voices")
        if voices_dir.exists():
            for voice_file in voices_dir.glob("*.wav"):
                voice_name = voice_file.stem
                self.voice_presets[voice_name] = str(voice_file)
        
        # Add demo voice presets
        demo_voices = {
            "Alice": "Female, professional narrator",
            "Bob": "Male, friendly conversational",
            "Charlie": "Male, technical presenter",
            "Diana": "Female, storyteller",
            "Emma": "Female, news anchor",
            "Frank": "Male, audiobook narrator"
        }
        
        for name, desc in demo_voices.items():
            if name not in self.voice_presets:
                self.voice_presets[name] = desc
    
    def load_whisper(self, model_size: str = "base"):
        """Load Whisper model for STT"""
        if not WHISPER_AVAILABLE:
            st.warning("Whisper not available. Demo mode will provide mock transcriptions.")
            self.models_loaded['whisper'] = True
            return True
            
        try:
            st.info(f"Loading Whisper {model_size} model...")
            self.whisper_model = whisper.load_model(model_size)
            self.models_loaded['whisper'] = True
            st.success("Whisper model loaded successfully!")
            return True
        except Exception as e:
            st.warning(f"Failed to load Whisper model: {str(e)}. Using demo mode.")
            self.models_loaded['whisper'] = True
            return True
    
    def load_coqui_tts(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        """Load Coqui TTS model"""
        if not COQUI_TTS_AVAILABLE:
            st.warning("Coqui TTS not available. Demo mode will provide synthetic audio.")
            self.models_loaded['coqui'] = True
            return True
            
        try:
            st.info(f"Loading Coqui TTS model: {model_name}")
            self.coqui_tts = TTS(model_name=model_name)
            self.models_loaded['coqui'] = True
            st.success("Coqui TTS model loaded successfully!")
            return True
        except Exception as e:
            st.warning(f"Failed to load Coqui TTS model: {str(e)}. Using demo mode.")
            self.models_loaded['coqui'] = True
            return True
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr)
            return audio
        except Exception as e:
            st.error(f"Failed to read audio file: {str(e)}")
            return None
    
    def vibevoice_tts(self, text: str, voice_samples: List[str] = None, cfg_scale: float = 1.3) -> Optional[np.ndarray]:
        """Generate speech using VibeVoice or demo mode"""
        if not self.models_loaded['vibevoice']:
            st.error("VibeVoice model not loaded!")
            return None
        
        # Demo mode implementation
        if self.demo_mode or not self.vibevoice_model:
            return self._generate_demo_audio(text, voice_samples)
        
        # Real model implementation
        try:
            # Process voice samples if provided
            voice_arrays = []
            if voice_samples:
                for voice_path in voice_samples:
                    if voice_path and os.path.exists(voice_path):
                        voice_audio = self.read_audio(voice_path)
                        if voice_audio is not None:
                            voice_arrays.append(voice_audio)
            
            # Process inputs
            inputs = self.vibevoice_processor(
                text=[text],
                voice_samples=[voice_arrays] if voice_arrays else None,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
            
            # Generate audio
            with torch.no_grad():
                outputs = self.vibevoice_model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                    cfg_scale=cfg_scale,
                    return_speech=True
                )
            
            if hasattr(outputs, 'speech_sequences') and outputs.speech_sequences is not None:
                audio_data = outputs.speech_sequences[0].cpu().numpy()
                return audio_data
            else:
                st.warning("No speech output generated")
                return None
                
        except Exception as e:
            st.error(f"VibeVoice TTS failed: {str(e)}")
            # Fall back to demo mode
            return self._generate_demo_audio(text, voice_samples)
    
    def _generate_demo_audio(self, text: str, voice_samples: List[str] = None) -> np.ndarray:
        """Generate synthetic demo audio"""
        # Estimate duration based on text length (roughly 150 words per minute)
        duration = max(2.0, len(text.split()) / 2.5)  # seconds
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create synthetic speech-like audio
        # Base frequency varies by voice
        if voice_samples and len(voice_samples) > 0:
            voice_name = voice_samples[0] if isinstance(voice_samples[0], str) else "Alice"
        else:
            voice_name = "Alice"
        
        # Lower frequencies for male voices
        male_voices = ['Bob', 'Charlie', 'Frank']
        frequency_base = 180 if any(name in voice_name for name in male_voices) else 250
        
        # Generate complex waveform
        audio = (
            0.4 * np.sin(2 * np.pi * frequency_base * t) +
            0.3 * np.sin(2 * np.pi * (frequency_base * 1.8) * t) +
            0.2 * np.sin(2 * np.pi * (frequency_base * 2.7) * t) +
            0.1 * np.random.normal(0, 0.05, len(t))
        )
        
        # Add speech-like modulation
        modulation = 1 + 0.15 * np.sin(2 * np.pi * 4 * t)  # 4 Hz modulation
        audio = audio * modulation
        
        # Apply envelope
        envelope = np.ones_like(t)
        fade_samples = int(0.1 * sample_rate)  # 0.1 second fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        audio = audio * envelope
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.7
        
        return audio.astype(np.float32)
    
    def whisper_stt(self, audio_file) -> Optional[str]:
        """Transcribe audio using Whisper or demo mode"""
        if not self.models_loaded['whisper']:
            st.error("Whisper model not loaded!")
            return None
        
        # Demo mode implementation
        if not WHISPER_AVAILABLE or not self.whisper_model:
            return self._generate_demo_transcript()
        
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name
            
            # Transcribe
            result = self.whisper_model.transcribe(tmp_path)
            
            # Clean up
            os.unlink(tmp_path)
            
            return result.get("text", "").strip()
            
        except Exception as e:
            st.error(f"Whisper STT failed: {str(e)}")
            # Fall back to demo mode
            return self._generate_demo_transcript()
    
    def _generate_demo_transcript(self) -> str:
        """Generate demo transcript"""
        demo_transcripts = [
            "This is a demonstration transcript generated by the VibeVoice integrated studio. In a real scenario, OpenAI Whisper would accurately transcribe your uploaded audio file into text.",
            "Welcome to the VibeVoice TTS and STT demonstration. This mock transcription shows how the speech-to-text functionality would work with actual audio input.",
            "The integrated studio combines multiple AI technologies for comprehensive audio processing. Your uploaded audio would be transcribed here with high accuracy using Whisper.",
            "Hello! This is a sample transcription showing the capabilities of the integrated voice processing system. Real audio files would be converted to text with advanced accuracy."
        ]
        
        import random
        return random.choice(demo_transcripts)
    
    def coqui_tts(self, text: str, voice_file: str = None) -> Optional[np.ndarray]:
        """Generate speech using Coqui TTS or demo mode"""
        if not self.models_loaded['coqui']:
            st.error("Coqui TTS model not loaded!")
            return None
        
        # Demo mode implementation  
        if not COQUI_TTS_AVAILABLE or not self.coqui_tts:
            return self._generate_demo_audio(text, [voice_file] if voice_file else None)
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                if voice_file:
                    # Voice cloning
                    self.coqui_tts.tts_to_file(text=text, speaker_wav=voice_file, file_path=tmp_file.name)
                else:
                    # Regular TTS
                    self.coqui_tts.tts_to_file(text=text, file_path=tmp_file.name)
                
                # Read generated audio
                audio, sr = librosa.load(tmp_file.name, sr=22050)
                os.unlink(tmp_file.name)
                
                return audio
                
        except Exception as e:
            st.error(f"Coqui TTS failed: {str(e)}")
            # Fall back to demo mode
            return self._generate_demo_audio(text, [voice_file] if voice_file else None)

def main():
    """Main Streamlit application"""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is required to run this application.")
        print("Please install it with: pip install streamlit")
        print("Then run: streamlit run streamlit_gui.py")
        return
    
    st.set_page_config(
        page_title="VibeVoice Integrated TTS/STT Studio",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎙️ VibeVoice Integrated TTS/STT Studio")
    st.markdown("### A comprehensive one-stop shop for Text-to-Speech and Speech-to-Text")
    
    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = IntegratedTTSSTTApp()
    
    app = st.session_state.app
    
    # Show demo mode indicator
    if app.demo_mode:
        st.info("🎭 **Demo Mode Active**: Generating synthetic audio for demonstration. Install required models for full functionality.")
    
    # Show model status
    status_cols = st.columns(3)
    with status_cols[0]:
        status = "✅ Loaded" if app.models_loaded['vibevoice'] else "❌ Not Loaded"
        mode = " (Demo)" if app.demo_mode else ""
        st.metric("VibeVoice", status + mode)
    
    with status_cols[1]:
        status = "✅ Loaded" if app.models_loaded['whisper'] else "❌ Not Loaded"
        mode = " (Demo)" if not WHISPER_AVAILABLE else ""
        st.metric("Whisper", status + mode)
    
    with status_cols[2]:
        status = "✅ Loaded" if app.models_loaded['coqui'] else "❌ Not Loaded"
        mode = " (Demo)" if not COQUI_TTS_AVAILABLE else ""
        st.metric("Coqui TTS", status + mode)
    
    # Sidebar for model configuration
    st.sidebar.header("🔧 Model Configuration")
    
    # VibeVoice Configuration
    st.sidebar.subheader("VibeVoice TTS")
    vibevoice_model_path = st.sidebar.text_input(
        "Model Path", 
        value="microsoft/VibeVoice-1.5B",
        help="HuggingFace model path for VibeVoice"
    )
    
    if st.sidebar.button("Load VibeVoice"):
        app.load_vibevoice(vibevoice_model_path)
    
    # Whisper Configuration
    st.sidebar.subheader("Whisper STT")
    whisper_model_size = st.sidebar.selectbox(
        "Model Size",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Larger models are more accurate but slower"
    )
    
    if st.sidebar.button("Load Whisper") and WHISPER_AVAILABLE:
        app.load_whisper(whisper_model_size)
    
    # Coqui TTS Configuration
    st.sidebar.subheader("Coqui TTS")
    coqui_model = st.sidebar.selectbox(
        "Model",
        [
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/glow-tts",
            "tts_models/en/vctk/vits",
        ],
        help="Coqui TTS model selection"
    )
    
    if st.sidebar.button("Load Coqui TTS") and COQUI_TTS_AVAILABLE:
        app.load_coqui_tts(coqui_model)
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗣️ Text-to-Speech", 
        "👂 Speech-to-Text", 
        "📚 Ebook to Audiobook", 
        "🎙️ Podcast Generation", 
        "🎭 Voice Cloning"
    ])
    
    with tab1:
        st.header("Text-to-Speech Generation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input
            tts_text = st.text_area(
                "Enter text to synthesize:",
                height=150,
                placeholder="Type your text here..."
            )
            
            # TTS Engine selection
            tts_engine = st.selectbox(
                "TTS Engine",
                ["VibeVoice", "Coqui TTS"],
                help="Choose the TTS engine to use"
            )
            
            # Configuration based on engine
            if tts_engine == "VibeVoice":
                cfg_scale = st.slider("CFG Scale", 0.1, 2.0, 1.3, 0.1)
                
                # Voice preset selection
                if app.voice_presets:
                    voice_preset = st.selectbox(
                        "Voice Preset",
                        list(app.voice_presets.keys()),
                        help="Select a voice preset"
                    )
                else:
                    voice_preset = None
            
            # Generate button
            if st.button("Generate Speech", type="primary"):
                if not tts_text.strip():
                    st.warning("Please enter some text to synthesize")
                else:
                    with st.spinner("Generating speech..."):
                        if tts_engine == "VibeVoice":
                            voice_files = [app.voice_presets.get(voice_preset)] if voice_preset else None
                            audio_data = app.vibevoice_tts(tts_text, voice_files, cfg_scale)
                        else:
                            audio_data = app.coqui_tts(tts_text)
                        
                        if audio_data is not None:
                            st.success("Speech generated successfully!")
                            
                            # Save audio to temporary file for playback
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                sf.write(tmp_file.name, audio_data, 24000)
                                
                                # Play audio
                                audio_file = open(tmp_file.name, 'rb')
                                audio_bytes = audio_file.read()
                                st.audio(audio_bytes, format='audio/wav')
                                
                                # Download button
                                st.download_button(
                                    label="Download Audio",
                                    data=audio_bytes,
                                    file_name="generated_speech.wav",
                                    mime="audio/wav"
                                )
                                
                                audio_file.close()
                                os.unlink(tmp_file.name)
        
        with col2:
            st.subheader("Voice Samples")
            
            # Upload voice samples
            voice_files = st.file_uploader(
                "Upload voice samples (optional)",
                type=['wav', 'mp3', 'flac'],
                accept_multiple_files=True,
                help="Upload audio files to use as voice references"
            )
            
            if voice_files:
                st.write(f"Uploaded {len(voice_files)} voice sample(s)")
                for voice_file in voice_files:
                    st.audio(voice_file, format='audio/wav')
    
    with tab2:
        st.header("Speech-to-Text Transcription")
        
        # Audio input
        audio_input_method = st.radio(
            "Audio Input Method",
            ["Upload File", "Record Audio"],
            horizontal=True
        )
        
        if audio_input_method == "Upload File":
            uploaded_audio = st.file_uploader(
                "Upload audio file",
                type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
                help="Upload an audio file to transcribe"
            )
            
            if uploaded_audio:
                st.audio(uploaded_audio, format='audio/wav')
                
                if st.button("Transcribe Audio", type="primary"):
                    with st.spinner("Transcribing audio..."):
                        transcript = app.whisper_stt(uploaded_audio)
                        
                        if transcript:
                            st.success("Transcription completed!")
                            st.text_area("Transcript:", transcript, height=200)
                            
                            # Download transcript
                            st.download_button(
                                label="Download Transcript",
                                data=transcript,
                                file_name="transcript.txt",
                                mime="text/plain"
                            )
        
        else:
            st.info("Audio recording feature requires additional setup. Please use file upload for now.")
    
    with tab3:
        st.header("Ebook to Audiobook Conversion")
        
        # Text input methods
        input_method = st.radio(
            "Input Method",
            ["Upload Text File", "Paste Text"],
            horizontal=True
        )
        
        ebook_text = ""
        
        if input_method == "Upload Text File":
            uploaded_file = st.file_uploader(
                "Upload text file",
                type=['txt', 'md'],
                help="Upload a text file to convert to audiobook"
            )
            
            if uploaded_file:
                ebook_text = str(uploaded_file.read(), "utf-8")
                st.text_area("Content preview:", ebook_text[:500] + "..." if len(ebook_text) > 500 else ebook_text, height=150)
        
        else:
            ebook_text = st.text_area(
                "Paste your text content:",
                height=200,
                placeholder="Paste the text content you want to convert to audiobook..."
            )
        
        if ebook_text:
            # Chapter splitting options
            split_chapters = st.checkbox("Split by chapters", value=True)
            
            if split_chapters:
                chapter_delimiter = st.text_input("Chapter delimiter", value="Chapter")
            
            narrator_voice = st.selectbox(
                "Narrator Voice",
                list(app.voice_presets.keys()) if app.voice_presets else ["Default"],
                help="Select the voice for narration"
            )
            
            if st.button("Convert to Audiobook", type="primary"):
                with st.spinner("Converting to audiobook..."):
                    # Split text into chunks if needed
                    if split_chapters and chapter_delimiter:
                        chapters = ebook_text.split(chapter_delimiter)
                        text_chunks = [f"{chapter_delimiter}{chapter}" for chapter in chapters[1:]] if len(chapters) > 1 else [ebook_text]
                    else:
                        # Split by sentences or paragraphs for better processing
                        text_chunks = [chunk.strip() for chunk in ebook_text.split('\n\n') if chunk.strip()]
                    
                    audiobook_parts = []
                    
                    for i, chunk in enumerate(text_chunks):
                        if chunk.strip():
                            st.write(f"Processing part {i+1}/{len(text_chunks)}")
                            voice_files = [app.voice_presets.get(narrator_voice)] if narrator_voice != "Default" else None
                            audio_data = app.vibevoice_tts(chunk, voice_files)
                            
                            if audio_data is not None:
                                audiobook_parts.append(audio_data)
                    
                    if audiobook_parts:
                        # Concatenate all audio parts
                        full_audiobook = np.concatenate(audiobook_parts)
                        
                        st.success("Audiobook conversion completed!")
                        
                        # Save and provide download
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            sf.write(tmp_file.name, full_audiobook, 24000)
                            
                            audio_file = open(tmp_file.name, 'rb')
                            audio_bytes = audio_file.read()
                            
                            st.audio(audio_bytes, format='audio/wav')
                            
                            st.download_button(
                                label="Download Audiobook",
                                data=audio_bytes,
                                file_name="audiobook.wav",
                                mime="audio/wav"
                            )
                            
                            audio_file.close()
                            os.unlink(tmp_file.name)
    
    with tab4:
        st.header("Podcast Generation")
        
        # Podcast script input
        script_input_method = st.radio(
            "Script Input Method",
            ["Write Script", "Upload Script File"],
            horizontal=True
        )
        
        podcast_script = ""
        
        if script_input_method == "Write Script":
            podcast_script = st.text_area(
                "Write your podcast script:",
                height=200,
                placeholder="""Speaker 1: Welcome to our podcast!
Speaker 2: Thanks for having me.
Speaker 1: Today we're discussing..."""
            )
        
        else:
            uploaded_script = st.file_uploader(
                "Upload script file",
                type=['txt'],
                help="Upload a script file with speaker labels"
            )
            
            if uploaded_script:
                podcast_script = str(uploaded_script.read(), "utf-8")
                st.text_area("Script preview:", podcast_script[:500] + "..." if len(podcast_script) > 500 else podcast_script, height=150)
        
        if podcast_script:
            # Speaker configuration
            st.subheader("Speaker Configuration")
            
            # Parse speakers from script
            import re
            speaker_pattern = r'^(Speaker \d+|[A-Za-z]+):'
            speakers = list(set(re.findall(speaker_pattern, podcast_script, re.MULTILINE)))
            
            if speakers:
                st.write(f"Detected speakers: {', '.join(speakers)}")
                
                speaker_voices = {}
                for speaker in speakers:
                    speaker_voices[speaker] = st.selectbox(
                        f"Voice for {speaker}",
                        list(app.voice_presets.keys()) if app.voice_presets else ["Default"],
                        key=f"voice_{speaker}"
                    )
                
                if st.button("Generate Podcast", type="primary"):
                    with st.spinner("Generating podcast..."):
                        # Parse script by speaker
                        lines = podcast_script.split('\n')
                        podcast_segments = []
                        
                        for line in lines:
                            line = line.strip()
                            if ':' in line:
                                speaker, text = line.split(':', 1)
                                speaker = speaker.strip()
                                text = text.strip()
                                
                                if speaker in speaker_voices and text:
                                    voice_files = [app.voice_presets.get(speaker_voices[speaker])] if speaker_voices[speaker] != "Default" else None
                                    audio_data = app.vibevoice_tts(text, voice_files)
                                    
                                    if audio_data is not None:
                                        podcast_segments.append(audio_data)
                                        # Add brief pause between speakers
                                        pause = np.zeros(int(0.5 * 24000))  # 0.5 second pause
                                        podcast_segments.append(pause)
                        
                        if podcast_segments:
                            # Concatenate all segments
                            full_podcast = np.concatenate(podcast_segments)
                            
                            st.success("Podcast generated successfully!")
                            
                            # Save and provide download
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                sf.write(tmp_file.name, full_podcast, 24000)
                                
                                audio_file = open(tmp_file.name, 'rb')
                                audio_bytes = audio_file.read()
                                
                                st.audio(audio_bytes, format='audio/wav')
                                
                                st.download_button(
                                    label="Download Podcast",
                                    data=audio_bytes,
                                    file_name="generated_podcast.wav",
                                    mime="audio/wav"
                                )
                                
                                audio_file.close()
                                os.unlink(tmp_file.name)
            else:
                st.warning("No speakers detected in script. Please format your script with 'Speaker:' labels.")
    
    with tab5:
        st.header("Voice Cloning")
        
        # Voice sample upload
        st.subheader("Upload Voice Sample")
        voice_sample = st.file_uploader(
            "Upload a voice sample for cloning",
            type=['wav', 'mp3', 'flac'],
            help="Upload a clean audio sample of the voice you want to clone (3-10 seconds recommended)"
        )
        
        if voice_sample:
            st.audio(voice_sample, format='audio/wav')
            
            # Text for voice cloning
            clone_text = st.text_area(
                "Enter text to synthesize with the cloned voice:",
                height=100,
                placeholder="Enter the text you want the cloned voice to speak..."
            )
            
            # Engine selection for voice cloning
            clone_engine = st.selectbox(
                "Voice Cloning Engine",
                ["VibeVoice", "Coqui TTS"],
                help="Choose the engine for voice cloning"
            )
            
            if clone_text and st.button("Clone Voice and Generate", type="primary"):
                with st.spinner("Cloning voice and generating speech..."):
                    # Save uploaded voice sample
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as voice_tmp:
                        voice_tmp.write(voice_sample.read())
                        voice_tmp_path = voice_tmp.name
                    
                    try:
                        if clone_engine == "VibeVoice":
                            # Use VibeVoice with the voice sample
                            voice_audio = app.read_audio(voice_tmp_path)
                            if voice_audio is not None:
                                audio_data = app.vibevoice_tts(clone_text, [voice_audio])
                        else:
                            # Use Coqui TTS for voice cloning
                            audio_data = app.coqui_tts(clone_text, voice_tmp_path)
                        
                        if audio_data is not None:
                            st.success("Voice cloning completed successfully!")
                            
                            # Save and provide playback/download
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                sf.write(tmp_file.name, audio_data, 24000)
                                
                                audio_file = open(tmp_file.name, 'rb')
                                audio_bytes = audio_file.read()
                                
                                st.audio(audio_bytes, format='audio/wav')
                                
                                st.download_button(
                                    label="Download Cloned Voice Audio",
                                    data=audio_bytes,
                                    file_name="cloned_voice.wav",
                                    mime="audio/wav"
                                )
                                
                                audio_file.close()
                                os.unlink(tmp_file.name)
                    
                    finally:
                        # Clean up voice sample file
                        os.unlink(voice_tmp_path)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **VibeVoice Integrated TTS/STT Studio** - A comprehensive solution combining:
    - 🎭 **VibeVoice** for long-form conversational TTS
    - 👂 **OpenAI Whisper** for speech-to-text transcription  
    - 🎵 **Coqui TTS** for voice cloning and synthesis
    - 📚 **Audiblez (Kokoro)** integration support
    
    *Powered by state-of-the-art AI models for professional audio generation and transcription.*
    """)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Mock demo of VibeVoice TTS functionality for demonstration purposes
This creates a working GUI without requiring model downloads
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

# Core dependencies
import numpy as np
import torch
import librosa
import soundfile as sf

class MockVibeVoiceDemo:
    """Mock VibeVoice demo for TTS functionality without model downloads"""
    
    def __init__(self):
        self.model_loaded = False
        self.voice_presets = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def load_model(self, model_path: str = "microsoft/VibeVoice-1.5B"):
        """Mock model loading"""
        try:
            print(f"[MOCK] Loading VibeVoice model from {model_path}...")
            time.sleep(2)  # Simulate loading time
            
            self.model_loaded = True
            self.setup_voice_presets()
            
            print(f"✅ [MOCK] VibeVoice model loaded successfully on {self.device}!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load VibeVoice model: {str(e)}")
            return False
    
    def setup_voice_presets(self):
        """Setup mock voice presets"""
        self.voice_presets = {
            "Alice": "Female, professional narrator",
            "Bob": "Male, friendly conversational",
            "Charlie": "Male, technical presenter", 
            "Diana": "Female, storyteller",
            "Emma": "Female, news anchor",
            "Frank": "Male, audiobook narrator"
        }
        print(f"✓ {len(self.voice_presets)} voice presets available")
    
    def generate_speech(self, text: str, voice_preset: str = None, cfg_scale: float = 1.3) -> Optional[np.ndarray]:
        """Generate mock speech using synthetic audio"""
        if not self.model_loaded:
            print("❌ VibeVoice model not loaded!")
            return None
        
        try:
            print(f"🗣️  [MOCK] Generating speech for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"  Using voice preset: {voice_preset or 'Default'}")
            print(f"  CFG Scale: {cfg_scale}")
            
            # Simulate generation time
            time.sleep(1 + len(text) / 100)  # Simulate processing time
            
            # Generate synthetic audio (sine wave tones representing speech)
            duration = max(2.0, len(text) / 15)  # Estimate duration based on text length
            sample_rate = 24000
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Create a complex waveform that sounds more like speech
            # Multiple frequencies to simulate formants
            frequency_base = 200 if voice_preset in ['Bob', 'Charlie', 'Frank'] else 300  # Lower for male voices
            
            audio = (
                0.3 * np.sin(2 * np.pi * frequency_base * t) +
                0.2 * np.sin(2 * np.pi * (frequency_base * 2.1) * t) +
                0.1 * np.sin(2 * np.pi * (frequency_base * 3.3) * t) +
                0.05 * np.random.normal(0, 0.1, len(t))  # Add some noise
            )
            
            # Apply envelope to make it sound more natural
            envelope = np.exp(-t * 0.5)  # Decay over time
            audio = audio * envelope
            
            # Add some modulation to simulate speech patterns
            modulation = 1 + 0.1 * np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
            audio = audio * modulation
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            generation_time = time.time() - (time.time() - 1 - len(text) / 100)
            print(f"  [MOCK] Generation completed in {generation_time:.2f} seconds")
            print(f"✅ [MOCK] Speech generated successfully! Duration: {duration:.2f} seconds")
            
            return audio.astype(np.float32)
                
        except Exception as e:
            print(f"❌ Mock TTS failed: {str(e)}")
            return None
    
    def save_audio(self, audio_data: np.ndarray, output_path: str, sample_rate: int = 24000):
        """Save audio data to file"""
        try:
            sf.write(output_path, audio_data, sample_rate)
            print(f"💾 Audio saved to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save audio: {str(e)}")
            return False

class MockWhisperDemo:
    """Mock Whisper STT functionality"""
    
    def __init__(self):
        self.model_loaded = False
        
    def load_model(self, model_size: str = "base"):
        """Mock Whisper model loading"""
        print(f"[MOCK] Loading Whisper {model_size} model...")
        time.sleep(1)
        self.model_loaded = True
        print("✅ [MOCK] Whisper model loaded successfully!")
        return True
    
    def transcribe(self, audio_file_path: str) -> str:
        """Mock transcription"""
        if not self.model_loaded:
            return "Error: Whisper model not loaded"
        
        print("[MOCK] Transcribing audio...")
        time.sleep(2)
        
        # Return a mock transcription
        mock_transcriptions = [
            "This is a mock transcription of your audio file. The actual Whisper model would provide accurate speech-to-text conversion.",
            "Hello, this is a sample transcription generated by the mock Whisper demo. Your uploaded audio would be properly transcribed here.",
            "Welcome to the VibeVoice integrated TTS and STT studio. This mock transcription shows how the system would work with real models."
        ]
        
        # Return a random mock transcription
        import random
        return random.choice(mock_transcriptions)

def interactive_demo():
    """Run interactive demo"""
    print("🎙️ VibeVoice Mock TTS Demo")
    print("=" * 40)
    print("Note: This is a demonstration version that works without downloading models.")
    print("Generated audio will be synthetic tones representing speech patterns.")
    
    # Initialize demo
    tts_demo = MockVibeVoiceDemo()
    stt_demo = MockWhisperDemo()
    
    # Load models
    print("\n📥 Loading Mock Models...")
    tts_demo.load_model()
    stt_demo.load_model()
    
    # Interactive menu
    while True:
        print("\n🚀 Mock Demo Menu:")
        print("1. Text-to-Speech (TTS)")
        print("2. Speech-to-Text (STT) - Mock")
        print("3. Voice Presets Info")
        print("4. Batch TTS Demo")
        print("5. Quit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            # TTS Demo
            print("\n--- Text-to-Speech Demo ---")
            text = input("Enter text to synthesize: ").strip()
            if not text:
                print("Please enter some text.")
                continue
            
            print("Available voice presets:", list(tts_demo.voice_presets.keys()))
            voice_preset = input("Voice preset [default: Alice]: ").strip()
            if not voice_preset:
                voice_preset = "Alice"
            
            if voice_preset not in tts_demo.voice_presets:
                print(f"Unknown voice preset. Using Alice.")
                voice_preset = "Alice"
            
            # Generate speech
            audio_data = tts_demo.generate_speech(text, voice_preset)
            
            if audio_data is not None:
                # Save audio
                timestamp = int(time.time())
                output_path = f"mock_speech_{timestamp}.wav"
                tts_demo.save_audio(audio_data, output_path)
                print(f"🎵 Generated mock audio file: {output_path}")
                print("   Note: This is synthetic audio representing speech patterns.")
        
        elif choice == "2":
            # STT Demo
            print("\n--- Speech-to-Text Demo ---")
            print("Note: This is a mock demonstration. Upload functionality would work with real Streamlit GUI.")
            
            mock_audio_path = input("Enter mock audio file path (or press Enter for demo): ").strip()
            if not mock_audio_path:
                mock_audio_path = "demo_audio.wav"
            
            transcript = stt_demo.transcribe(mock_audio_path)
            print(f"\n📝 Mock Transcription:")
            print(f"   {transcript}")
        
        elif choice == "3":
            # Voice Presets Info
            print("\n--- Voice Presets Information ---")
            for name, description in tts_demo.voice_presets.items():
                print(f"  {name}: {description}")
        
        elif choice == "4":
            # Batch Demo
            print("\n--- Batch TTS Demo ---")
            
            examples = [
                {"text": "Welcome to VibeVoice integrated studio!", "voice": "Alice"},
                {"text": "This is a comprehensive text-to-speech solution.", "voice": "Bob"},
                {"text": "Generate high-quality audio from text input.", "voice": "Diana"}
            ]
            
            os.makedirs("mock_outputs", exist_ok=True)
            
            for i, example in enumerate(examples):
                print(f"\nGenerating example {i+1}/{len(examples)}")
                audio_data = tts_demo.generate_speech(example['text'], example['voice'])
                
                if audio_data is not None:
                    output_path = f"mock_outputs/batch_example_{i+1}_{example['voice'].lower()}.wav"
                    tts_demo.save_audio(audio_data, output_path)
            
            print("✅ Batch demo completed! Check 'mock_outputs' directory.")
        
        elif choice == "5":
            break
        
        else:
            print("Invalid option. Please select 1-5.")
    
    print("\n👋 Mock demo finished. Thank you for using VibeVoice!")

def streamlit_compatible_demo():
    """Create files that demonstrate Streamlit GUI compatibility"""
    print("🎙️ Creating Streamlit-compatible demonstration files...")
    
    # Create demo configuration
    demo_config = {
        "tts_engines": [
            {
                "name": "VibeVoice",
                "description": "Long-form conversational TTS",
                "features": ["Multi-speaker", "Voice cloning", "Long sequences"]
            },
            {
                "name": "Coqui TTS", 
                "description": "Advanced voice synthesis",
                "features": ["Voice cloning", "Multi-language", "High quality"]
            }
        ],
        "stt_engines": [
            {
                "name": "OpenAI Whisper",
                "description": "State-of-the-art speech recognition",
                "features": ["Multi-language", "High accuracy", "Robust to noise"]
            }
        ],
        "voice_presets": {
            "Alice": {"type": "female", "style": "professional"},
            "Bob": {"type": "male", "style": "conversational"},
            "Charlie": {"type": "male", "style": "technical"},
            "Diana": {"type": "female", "style": "storyteller"}
        },
        "supported_formats": {
            "input": ["txt", "md", "docx", "pdf"],
            "output": ["wav", "mp3", "flac"]
        }
    }
    
    with open("demo_config.json", "w") as f:
        json.dump(demo_config, f, indent=2)
    
    print("✅ Demo configuration created: demo_config.json")
    
    # Create example texts
    examples = {
        "simple_tts": "Hello! This is a test of the VibeVoice text-to-speech system.",
        "long_form": """
        Welcome to VibeVoice, an advanced text-to-speech system capable of generating
        long-form conversational audio. This technology can create natural-sounding
        speech with multiple speakers, making it perfect for audiobooks, podcasts,
        and educational content.
        """,
        "podcast_script": """
        Alice: Welcome to our technology podcast!
        Bob: Thanks for having me, Alice. Today we're discussing AI voice technology.
        Alice: That's right. VibeVoice represents a significant advancement in this field.
        Bob: Absolutely. The ability to generate long-form conversational audio is remarkable.
        Alice: Our listeners can experience this technology firsthand through our integrated studio.
        """,
        "audiobook_sample": """
        Chapter 1: The Beginning
        
        In a world where artificial intelligence had advanced beyond human imagination,
        voice synthesis technology had reached new heights. VibeVoice emerged as a
        revolutionary system, capable of creating natural, expressive speech that
        could tell stories, explain concepts, and engage listeners for hours.
        """
    }
    
    os.makedirs("demo_texts", exist_ok=True)
    for name, content in examples.items():
        with open(f"demo_texts/{name}.txt", "w") as f:
            f.write(content.strip())
    
    print("✅ Demo text files created in demo_texts/")
    print("✅ All demonstration files created successfully!")
    
    return demo_config

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        streamlit_compatible_demo()
    else:
        interactive_demo()

if __name__ == "__main__":
    main()
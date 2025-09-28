#!/usr/bin/env python3
"""
Simple demo of VibeVoice TTS functionality without external dependencies
This can be run immediately after installing VibeVoice core dependencies
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time

# Core dependencies
import numpy as np
import torch
import librosa
import soundfile as sf

# VibeVoice imports
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

class SimpleVibeVoiceDemo:
    """Simple VibeVoice demo for TTS functionality"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.voice_presets = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def load_model(self, model_path: str = "microsoft/VibeVoice-1.5B"):
        """Load VibeVoice model and processor"""
        try:
            print(f"Loading VibeVoice model from {model_path}...")
            
            # Load processor
            self.processor = VibeVoiceProcessor.from_pretrained(model_path)
            print("✓ Processor loaded")
            
            # Load model
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            print("✓ Model loaded")
            
            # Setup voice presets
            self.setup_voice_presets()
            
            print(f"✅ VibeVoice model loaded successfully on {self.device}!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load VibeVoice model: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory"""
        voices_dir = Path("demo/voices")
        if voices_dir.exists():
            for voice_file in voices_dir.glob("*.wav"):
                voice_name = voice_file.stem
                self.voice_presets[voice_name] = str(voice_file)
                print(f"  Found voice preset: {voice_name}")
        
        # Add some default voice names if no files found
        if not self.voice_presets:
            self.voice_presets = {
                "Alice": None,
                "Bob": None,
                "Charlie": None,
                "Diana": None
            }
            print("  Using default voice presets")
        
        print(f"✓ {len(self.voice_presets)} voice presets available")
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr)
            return audio
        except Exception as e:
            print(f"❌ Failed to read audio file: {str(e)}")
            return None
    
    def generate_speech(self, text: str, voice_preset: str = None, cfg_scale: float = 1.3) -> Optional[np.ndarray]:
        """Generate speech using VibeVoice"""
        if not self.model or not self.processor:
            print("❌ VibeVoice model not loaded!")
            return None
        
        try:
            print(f"🗣️  Generating speech for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Process voice samples if provided
            voice_arrays = []
            if voice_preset and voice_preset in self.voice_presets:
                voice_path = self.voice_presets[voice_preset]
                if voice_path and os.path.exists(voice_path):
                    voice_audio = self.read_audio(voice_path)
                    if voice_audio is not None:
                        voice_arrays.append(voice_audio)
                        print(f"  Using voice preset: {voice_preset}")
            
            # Process inputs
            inputs = self.processor(
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
            print("  Generating audio...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                    cfg_scale=cfg_scale,
                    return_speech=True
                )
            
            generation_time = time.time() - start_time
            print(f"  Generation completed in {generation_time:.2f} seconds")
            
            if hasattr(outputs, 'speech_sequences') and outputs.speech_sequences is not None:
                audio_data = outputs.speech_sequences[0].cpu().numpy()
                print(f"✅ Speech generated successfully! Duration: {len(audio_data)/24000:.2f} seconds")
                return audio_data
            else:
                print("⚠️  No speech output generated")
                return None
                
        except Exception as e:
            print(f"❌ VibeVoice TTS failed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
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

def interactive_demo():
    """Run interactive demo"""
    print("🎙️ VibeVoice Simple TTS Demo")
    print("=" * 40)
    
    # Initialize demo
    demo = SimpleVibeVoiceDemo()
    
    # Load model
    print("\n📥 Loading VibeVoice model...")
    if not demo.load_model():
        print("❌ Failed to load model. Exiting.")
        return
    
    # Interactive loop
    print("\n🚀 Demo ready! Enter text to synthesize (or 'quit' to exit)")
    print("Available voice presets:", list(demo.voice_presets.keys()))
    
    while True:
        print("\n" + "-" * 40)
        
        # Get text input
        text = input("Enter text to synthesize: ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        # Get voice preset
        voice_preset = input(f"Voice preset ({'/'.join(demo.voice_presets.keys())}) [default: Alice]: ").strip()
        if not voice_preset:
            voice_preset = "Alice"
        
        if voice_preset not in demo.voice_presets:
            print(f"Unknown voice preset. Using Alice.")
            voice_preset = "Alice"
        
        # Generate speech
        audio_data = demo.generate_speech(text, voice_preset)
        
        if audio_data is not None:
            # Save audio
            timestamp = int(time.time())
            output_path = f"output_speech_{timestamp}.wav"
            demo.save_audio(audio_data, output_path)
            
            print(f"🎵 Generated audio file: {output_path}")
            print("   You can play this file with any audio player.")
        else:
            print("❌ Failed to generate speech.")
    
    print("\n👋 Demo finished. Thank you for using VibeVoice!")

def batch_demo():
    """Run batch demo with example texts"""
    print("🎙️ VibeVoice Batch TTS Demo")
    print("=" * 40)
    
    # Initialize demo
    demo = SimpleVibeVoiceDemo()
    
    # Load model
    print("\n📥 Loading VibeVoice model...")
    if not demo.load_model():
        print("❌ Failed to load model. Exiting.")
        return
    
    # Example texts
    examples = [
        {
            "text": "Hello! Welcome to VibeVoice, an advanced text-to-speech system.",
            "voice": "Alice",
            "description": "Simple greeting"
        },
        {
            "text": "VibeVoice can generate long-form conversational audio with multiple speakers and natural turn-taking.",
            "voice": "Bob", 
            "description": "Technical description"
        },
        {
            "text": "This technology opens up exciting possibilities for audiobook creation, podcast generation, and voice cloning applications.",
            "voice": "Charlie",
            "description": "Applications overview"
        }
    ]
    
    print(f"\n🚀 Running batch demo with {len(examples)} examples...")
    
    os.makedirs("demo_outputs", exist_ok=True)
    
    for i, example in enumerate(examples):
        print(f"\n📝 Example {i+1}/{len(examples)}: {example['description']}")
        print(f"   Text: {example['text'][:60]}{'...' if len(example['text']) > 60 else ''}")
        print(f"   Voice: {example['voice']}")
        
        # Generate speech
        audio_data = demo.generate_speech(example['text'], example['voice'])
        
        if audio_data is not None:
            # Save audio
            output_path = f"demo_outputs/example_{i+1}_{example['voice'].lower()}.wav"
            demo.save_audio(audio_data, output_path)
        else:
            print(f"❌ Failed to generate example {i+1}")
    
    print(f"\n✅ Batch demo completed! Check the 'demo_outputs' directory for generated audio files.")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        batch_demo()
    else:
        interactive_demo()

if __name__ == "__main__":
    main()
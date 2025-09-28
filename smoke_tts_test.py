#!/usr/bin/env python3
"""
Quick TTS smoke test - generates audio using available engines
"""

import os
from tts_backend import generate_speech_simple, list_available_voices

def main():
    print("🎭 TTS Smoke Test")
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # List available voices
    voices = list_available_voices()
    print(f"📊 Found {len(voices)} voices:")
    for i, (key, info) in enumerate(list(voices.items())[:5], 1):
        print(f"  {i}. {key} ({info['engine']}) - {info['description'][:50]}...")
    
    if not voices:
        print("❌ No voices available")
        return False
    
    # Test with first available voice
    voice_key = list(voices.keys())[0]
    voice_info = voices[voice_key]
    
    print(f"\n🎙️ Testing voice: {voice_key}")
    print(f"   Engine: {voice_info['engine']}")
    print(f"   Description: {voice_info['description']}")
    
    # Generate test audio
    test_text = "Hello, this is a quick TTS smoke test. The system is working correctly."
    output_path = "outputs/smoke_test.wav"
    
    print(f"\n🔊 Generating: '{test_text[:30]}...'")
    
    success = generate_speech_simple(
        text=test_text,
        voice_name=voice_key,
        output_path=output_path
    )
    
    if success and os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"✅ Audio generated successfully!")
        print(f"   File: {output_path}")
        print(f"   Size: {file_size:,} bytes")
        
        # Basic validation
        if file_size > 1000:  # Reasonable audio file size
            print("✅ File size looks reasonable for audio")
        else:
            print("⚠️ File size seems small - might be placeholder/silence")
            
        return True
    else:
        print(f"❌ Audio generation failed")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ Smoke test PASSED' if success else '❌ Smoke test FAILED'}")

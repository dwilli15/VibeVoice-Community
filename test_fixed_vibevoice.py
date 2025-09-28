#!/usr/bin/env python3
"""
Test the fixed VibeVoice TTS generation
"""

print("=== TESTING FIXED VIBEVOICE TTS ===")

try:
    from tts_backend import MultiModelTTSBackend, TTSRequest
    import tempfile
    
    backend = MultiModelTTSBackend()
    
    # Find a VibeVoice voice
    vibevoice_voices = [v for v in backend.voices.values() if v.engine.value == "vibevoice"]
    
    if not vibevoice_voices:
        print("❌ No VibeVoice voices found")
        exit(1)
    
    test_voice = vibevoice_voices[0]
    print(f"Testing with voice: {test_voice.name}")
    
    # Create a test request
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        output_path = f.name
    
    request = TTSRequest(
        text="Hello, this is a test of VibeVoice speech generation.",
        voice=test_voice,
        output_path=output_path
    )
    
    print("Generating speech...")
    success = backend.generate_speech(request)
    
    if success:
        print(f"✅ Speech generation successful!")
        print(f"Output file: {output_path}")
        
        # Check file size
        import os
        size = os.path.getsize(output_path)
        print(f"File size: {size} bytes")
        
        if size > 1000:  # Should be more than 1KB for real audio
            print("✅ Audio file appears to contain real data")
        else:
            print("⚠️ Audio file is very small - might still be placeholder")
    else:
        print("❌ Speech generation failed")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("=== TEST COMPLETE ===")

#!/usr/bin/env python3
"""
Test VibeVoice functionality specifically
"""

print("=== VIBEVOICE SPECIFIC TESTS ===")

# Test 1: VibeVoice imports
print("\n1. Testing VibeVoice imports...")
try:
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    print("   ✓ VibeVoiceConfig import OK")
except Exception as e:
    print(f"   ✗ VibeVoiceConfig import failed: {e}")

try:
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    print("   ✓ VibeVoiceForConditionalGenerationInference import OK")
except Exception as e:
    print(f"   ✗ VibeVoiceForConditionalGenerationInference import failed: {e}")

try:
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    print("   ✓ VibeVoiceProcessor import OK")
except Exception as e:
    print(f"   ✗ VibeVoiceProcessor import failed: {e}")

# Test 2: Model availability
print("\n2. Testing model availability...")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name()}")
except Exception as e:
    print(f"   ✗ PyTorch check failed: {e}")

# Test 3: VibeVoice model loading
print("\n3. Testing VibeVoice model loading...")
try:
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    
    model_path = "microsoft/VibeVoice-1.5B"
    print(f"   Loading processor for: {model_path}")
    
    processor = VibeVoiceProcessor.from_pretrained(model_path)
    print("   ✓ Processor loaded successfully")
    
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    
    print(f"   Loading model for: {model_path}")
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("   ✓ Model loaded successfully")
    
    # Test 4: Simple generation
    print("\n4. Testing speech generation...")
    test_text = "Hello, this is a test."
    print(f"   Generating speech for: {test_text}")
    
    # Process text
    inputs = processor(test_text, return_tensors="pt")
    print("   ✓ Text processed")
    
    # Generate speech
    with torch.no_grad():
        audio_output = model.generate(**inputs, max_length=1000)
    print("   ✓ Audio generated")
    print(f"   Audio shape: {audio_output.shape}")
    
except Exception as e:
    print(f"   ✗ VibeVoice generation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: TTS Backend integration
print("\n5. Testing TTS Backend VibeVoice integration...")
try:
    from tts_backend import MultiModelTTSBackend
    backend = MultiModelTTSBackend()
    
    print(f"   Backend voices: {len(backend.voices)}")
    vibevoice_voices = [v for v in backend.voices.keys() if 'VibeVoice' in v or any(x in v for x in ['bf_', 'af_', 'en-'])]
    print(f"   VibeVoice voices: {vibevoice_voices}")
    
    if vibevoice_voices:
        test_voice = vibevoice_voices[0]
        voice_obj = backend.voices[test_voice]
        print(f"   Test voice: {test_voice}")
        print(f"   Voice engine: {voice_obj.engine}")
        print(f"   Voice model: {getattr(voice_obj, 'model', 'N/A')}")
    
except Exception as e:
    print(f"   ✗ TTS Backend test failed: {e}")

print("\n=== VIBEVOICE TESTS COMPLETE ===")

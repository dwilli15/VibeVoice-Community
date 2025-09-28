#!/usr/bin/env python3
"""
Debug VibeVoice generation step by step
"""

print("=== DEBUGGING VIBEVOICE GENERATION ===")

try:
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    import torch
    
    # Load processor and model
    model_path = "microsoft/VibeVoice-1.5B"
    print(f"Loading processor from: {model_path}")
    processor = VibeVoiceProcessor.from_pretrained(model_path)
    
    print(f"Loading model from: {model_path}")
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Test text processing
    test_text = "Speaker 1: Hello, this is a test."
    print(f"Processing text: {test_text}")
    
    inputs = processor(test_text, return_tensors="pt")
    print(f"Processed inputs keys: {list(inputs.keys())}")
    
    # Check inputs structure
    for key, value in inputs.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: {type(value)} - {value}")
    
    # Test generation
    print("Attempting generation...")
    device = next(model.parameters()).device
    print(f"Model device: {device}")
    
    # Move inputs to device
    inputs_on_device = {}
    for k, v in inputs.items():
        if hasattr(v, 'to'):
            inputs_on_device[k] = v.to(device)
        else:
            inputs_on_device[k] = v
    
    # Check model config
    print(f"Model config: {model.config}")
    print(f"Model generation config: {getattr(model, 'generation_config', 'None')}")
    
    # Try generation with explicit parameters
    with torch.no_grad():
        try:
            output = model.generate(
                **inputs_on_device,
                max_length=1000,
                do_sample=True,
                temperature=0.8,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=getattr(processor.tokenizer, 'bos_token_id', None)
            )
            print(f"Generation successful! Output shape: {output.shape}")
        except Exception as e:
            print(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
    
except Exception as e:
    print(f"Debug failed: {e}")
    import traceback
    traceback.print_exc()

print("=== DEBUG COMPLETE ===")

#!/usr/bin/env python3
"""
Test using the external tokenizer to fix VibeVoice
"""

from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
import torch

print('=== TESTING WITH EXTERNAL TOKENIZER ===')

# Try loading with the external tokenizer path
try:
    tokenizer_path = 'D:/omen/temp/tokenizer'
    model_path = 'microsoft/VibeVoice-1.5B'
    
    print(f'Loading processor with external tokenizer: {tokenizer_path}')
    
    # Try different ways to load with external tokenizer
    try:
        # Method 1: Pass tokenizer path as parameter
        processor = VibeVoiceProcessor.from_pretrained(
            model_path,
            tokenizer_path=tokenizer_path
        )
        print('✓ Method 1: Processor loaded with external tokenizer')
    except Exception as e1:
        print(f'Method 1 failed: {e1}')
        try:
            # Method 2: Load tokenizer separately
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            processor = VibeVoiceProcessor.from_pretrained(model_path)
            processor.tokenizer = tokenizer
            print('✓ Method 2: Processor loaded with external tokenizer')
        except Exception as e2:
            print(f'Method 2 failed: {e2}')
            # Method 3: Default loading
            processor = VibeVoiceProcessor.from_pretrained(model_path)
            print('✓ Method 3: Default processor loaded')
    
    # Load model
    print(f'Loading model: {model_path}')
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    print('✓ Model loaded')
    
    # Test processing
    test_text = 'Speaker 1: Hello, this is a test.'
    print(f'Processing: {test_text}')
    
    inputs = processor(test_text, return_tensors='pt')
    print('✓ Text processed successfully')
    print(f'Input keys: {list(inputs.keys())}')
    
    # Check tokenizer attributes
    print(f'Tokenizer type: {type(processor.tokenizer)}')
    print(f'Has bos_token_id: {hasattr(processor.tokenizer, "bos_token_id")}')
    print(f'Has eos_token_id: {hasattr(processor.tokenizer, "eos_token_id")}')
    print(f'Has pad_token_id: {hasattr(processor.tokenizer, "pad_token_id")}')
    
    if hasattr(processor.tokenizer, 'bos_token_id'):
        print(f'bos_token_id: {processor.tokenizer.bos_token_id}')
    if hasattr(processor.tokenizer, 'eos_token_id'):
        print(f'eos_token_id: {processor.tokenizer.eos_token_id}')
    if hasattr(processor.tokenizer, 'pad_token_id'):
        print(f'pad_token_id: {processor.tokenizer.pad_token_id}')
    
    # Test generation with proper token IDs
    print('\nTesting generation...')
    device = next(model.parameters()).device
    inputs_on_device = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    # Get proper token IDs
    bos_id = getattr(processor.tokenizer, 'bos_token_id', None)
    eos_id = getattr(processor.tokenizer, 'eos_token_id', None)
    pad_id = getattr(processor.tokenizer, 'pad_token_id', None)
    
    print(f'Using token IDs - bos: {bos_id}, eos: {eos_id}, pad: {pad_id}')
    
    with torch.no_grad():
        output = model.generate(
            **inputs_on_device,
            max_length=1000,
            do_sample=True,
            temperature=0.8,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            bos_token_id=bos_id
        )
        print(f'✅ Generation successful! Output shape: {output.shape}')

except Exception as e:
    print(f'❌ Failed: {e}')
    import traceback
    traceback.print_exc()

print('=== TEST COMPLETE ===')

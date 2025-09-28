# -*- coding: utf-8 -*-
import sys
import traceback
import logging
import torch
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vibevoice_generation():
    """Test VibeVoice generation with detailed error tracking"""
    try:
        # Check PyTorch and CUDA
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Compute capability: {torch.cuda.get_device_capability(0)}")
            logger.info(f"Supported architectures: {torch.cuda.get_arch_list()}")

        # Try importing VibeVoice
        try:
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
            logger.info("✅ VibeVoice imports successful")
        except Exception as e:
            logger.error(f"❌ VibeVoice import failed: {e}")
            return

        # Set up model path and voice
        model_path = "microsoft/VibeVoice-1.5b"
        voice_file = "./demo/voices/en-Alice_woman.wav"
        
        if not Path(voice_file).exists():
            logger.error(f"❌ Voice file not found: {voice_file}")
            return

        # Test text
        text = "Speaker 1: Hello, this is a test."
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load processor
        logger.info("Loading processor...")
        processor = VibeVoiceProcessor.from_pretrained(model_path)
        logger.info("✅ Processor loaded")
        
        # Load model
        logger.info("Loading model...")
        if device == "cuda":
            try:
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",
                    attn_implementation="flash_attention_2",
                )
            except Exception as fa_err:
                logger.warning(f"Flash Attention 2 failed, using SDPA: {fa_err}")
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",
                    attn_implementation="sdpa",
                )
        else:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                attn_implementation="sdpa",
            )
        
        model.eval()
        if hasattr(model, "set_ddpm_inference_steps"):
            model.set_ddpm_inference_steps(num_steps=8)
        logger.info("✅ Model loaded")
        
        # Prepare inputs
        logger.info("Preparing inputs...")
        inputs = processor(
            text=[text],
            voice_samples=[[str(voice_file)]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        logger.info(f"Input keys: {list(inputs.keys())}")
        
        # Move inputs to device
        logger.info("Moving inputs to device...")
        for k, v in inputs.items():
            if torch.is_tensor(v):
                logger.info(f"Moving {k} (type: {type(v)}) to {device}")
                if v is not None:
                    inputs[k] = v.to(device)
                else:
                    logger.warning(f"Input {k} is None!")
        
        # Generate
        logger.info("Starting generation...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.3,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
        )
        logger.info(f"✅ Generation completed. Outputs type: {type(outputs)}")
        
        if hasattr(outputs, 'speech_outputs'):
            logger.info(f"Speech outputs: {len(outputs.speech_outputs) if outputs.speech_outputs else 'None'}")
            if outputs.speech_outputs:
                speech_output = outputs.speech_outputs[0]
                logger.info(f"First speech output type: {type(speech_output)}")
                logger.info(f"First speech output shape: {speech_output.shape if hasattr(speech_output, 'shape') else 'No shape'}")
        else:
            logger.warning("❌ No speech_outputs attribute in outputs")
            
        # Save audio
        output_path = "test_output.wav"
        logger.info(f"Saving audio to {output_path}...")
        processor.save_audio(
            outputs.speech_outputs[0],
            output_path=output_path,
        )
        logger.info("✅ Audio saved successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {e}")
        logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    test_vibevoice_generation()
# -*- coding: utf-8 -*-
"""
Simplified TTS Backend for Ebook Conversion
Works with existing VibeVoice setup and provides fallback options.

Notes:
    Added explicit UTF-8 coding cookie and text normalization to avoid
    Windows console mojibake (e.g. RIGHT SINGLE QUOTATION MARK showing as â€™)
    which previously surfaced as a misleading SyntaxError on the f-string
    constructing the script line.
"""

import os
import sys
import json
import tempfile
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# VibeVoice imports (lazy)
try:
    import torch  # for device detection
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    VIBEVOICE_IMPORTS_OK = True
except Exception:
    VIBEVOICE_IMPORTS_OK = False

# Audio processing
try:
    import soundfile as sf
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# TTS engines (optional)
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    pyttsx3 = None

@dataclass
class Voice:
    """Simple voice representation"""
    name: str
    language: str
    gender: str
    engine: str
    file_path: Optional[str] = None  # Path to custom voice file
    created_at: Optional[str] = None  # Timestamp when added
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Voice':
        """Create Voice from dictionary"""
        return cls(**data)

class SimpleTTSBackend:
    """Simplified TTS backend for ebook conversion with voice persistence"""
    
    def __init__(self):
        self.logger = logging.getLogger('SimpleTTS')
        self.voices = self._discover_voices()
        self.temp_dir = Path(tempfile.gettempdir()) / "vibevoice_tts"
        self.temp_dir.mkdir(exist_ok=True)
        # Cache for VibeVoice to avoid reloading per chunk
        self._vv_model = None
        self._vv_processor = None
        self._vv_device = None
        self._vv_model_path = None
        
        # Initialize custom voices directory and config
        self.custom_voices_dir = self._get_custom_voices_dir()
        self.config_file = self.custom_voices_dir / "voice_library.json"
        
        # Load custom voices from persistent storage
        self._load_custom_voices()
        
        # Check for available TTS engines
        self.vibevoice_available = self._check_vibevoice()
        self.espeak_available = self._check_espeak()
        self.pyttsx3_available = self._check_pyttsx3()
        
        self.logger.info(f"TTS Backend initialized:")
        self.logger.info(f"  VibeVoice: {'✅' if self.vibevoice_available else '❌'} (Primary)")
        self.logger.info(f"  eSpeak: {'✅' if self.espeak_available else '❌'} (Fallback)")
        self.logger.info(f"  pyttsx3: {'✅' if self.pyttsx3_available else '❌'} (Fallback)")
        
        # Report default voice
        default_voice = self.get_default_voice()
        if default_voice:
            self.logger.info(f"  Default Voice: {default_voice.name} ({default_voice.engine})")
    
    def _check_vibevoice(self) -> bool:
        """Check if VibeVoice is available"""
        try:
            # Check for gradio demo
            demo_path = Path("demo/gradio_demo.py")
            return demo_path.exists()
        except:
            return False
    
    def _check_espeak(self) -> bool:
        """Check if eSpeak is available"""
        try:
            subprocess.run(["espeak", "--version"], 
                         capture_output=True, check=True)
            return True
        except:
            return False
    
    def _check_pyttsx3(self) -> bool:
        """Check if pyttsx3 is available"""
        return PYTTSX3_AVAILABLE
    
    def _discover_voices(self) -> List[Voice]:
        """Discover available voices (VibeVoice prioritized)"""
        voices = []
        
        # VibeVoice voices FIRST (primary engine for this project)
        vibevoice_voices = [
            Voice("bf_isabella", "en", "female", "vibevoice"),  # Default voice
            Voice("af_heart", "en", "female", "vibevoice"),
            Voice("en-Alice_woman", "en", "female", "vibevoice"),
            Voice("en-Carter_man", "en", "male", "vibevoice"),
            Voice("en-Frank_man", "en", "male", "vibevoice"),
            Voice("en-Maya_woman", "en", "female", "vibevoice"),
            Voice("zh-Anchen_man", "zh", "male", "vibevoice"),
            Voice("zh-Bowen_man", "zh", "male", "vibevoice"),
            Voice("zh-Xinran_woman", "zh", "female", "vibevoice"),
        ]
        voices.extend(vibevoice_voices)
        
        # eSpeak voices (fallback if VibeVoice unavailable)
        if self._check_espeak():
            espeak_voices = [
                Voice("en-us", "en", "neutral", "espeak"),
                Voice("en-gb", "en", "neutral", "espeak"),
                Voice("fr", "fr", "neutral", "espeak"),
                Voice("de", "de", "neutral", "espeak"),
                Voice("es", "es", "neutral", "espeak"),
            ]
            voices.extend(espeak_voices)
        
        return voices
    
    def get_voices(self) -> List[Voice]:
        """Get list of available voices"""
        return self.voices
    
    def get_default_voice(self) -> Optional[Voice]:
        """Get the default voice (VibeVoice preferred)"""
        # Prefer VibeVoice voices first
        vibevoice_voices = [v for v in self.voices if v.engine == "vibevoice"]
        if vibevoice_voices:
            return vibevoice_voices[0]  # bf_isabella by default
        
        # Fallback to first available voice
        return self.voices[0] if self.voices else None
    
    def _get_custom_voices_dir(self) -> Path:
        """Get directory for storing custom voices"""
        # Use user's AppData directory on Windows
        user_data = Path.home() / "AppData" / "Local" / "VibeVoice-Community" / "voices"
        user_data.mkdir(parents=True, exist_ok=True)
        return user_data
    
    def _load_custom_voices(self):
        """Load custom voices from persistent storage"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    custom_voices_data = json.load(f)
                
                for voice_data in custom_voices_data:
                    # Verify the voice file still exists
                    if voice_data.get('file_path') and Path(voice_data['file_path']).exists():
                        voice = Voice.from_dict(voice_data)
                        # Add to voices list if not already present
                        if not any(v.name == voice.name for v in self.voices):
                            self.voices.append(voice)
                            self.logger.info(f"Loaded custom voice: {voice.name}")
                    else:
                        self.logger.warning(f"Custom voice file missing: {voice_data.get('file_path')}")
        except Exception as e:
            self.logger.error(f"Failed to load custom voices: {e}")
    
    def save_custom_voice(self, voice_file_path: str, voice_name: str, 
                         language: str = "en", gender: str = "neutral") -> bool:
        """Save a custom voice file for future use"""
        try:
            source_path = Path(voice_file_path)
            if not source_path.exists():
                self.logger.error(f"Voice file not found: {voice_file_path}")
                return False
            
            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in voice_name if c.isalnum() or c in ('-', '_'))
            file_extension = source_path.suffix
            new_filename = f"{safe_name}_{timestamp}{file_extension}"
            
            # Copy to custom voices directory
            dest_path = self.custom_voices_dir / new_filename
            shutil.copy2(source_path, dest_path)
            
            # Create voice object
            voice = Voice(
                name=voice_name,
                language=language,
                gender=gender,
                engine="custom",
                file_path=str(dest_path),
                created_at=datetime.now().isoformat()
            )
            
            # Add to current session
            self.voices.append(voice)
            
            # Save to persistent storage
            self._save_voice_library()
            
            self.logger.info(f"Custom voice saved: {voice_name} -> {dest_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save custom voice: {e}")
            return False
    
    def _save_voice_library(self):
        """Save current custom voices to JSON file"""
        try:
            # Only save custom voices (not built-in ones)
            custom_voices = [v for v in self.voices if v.engine == "custom"]
            voice_data = [voice.to_dict() for voice in custom_voices]
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(voice_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved {len(custom_voices)} custom voices to library")
            
        except Exception as e:
            self.logger.error(f"Failed to save voice library: {e}")
    
    def remove_custom_voice(self, voice_name: str) -> bool:
        """Remove a custom voice from storage"""
        try:
            # Find the voice
            voice = next((v for v in self.voices if v.name == voice_name and v.engine == "custom"), None)
            if not voice:
                self.logger.warning(f"Custom voice not found: {voice_name}")
                return False
            
            # Remove file if it exists
            if voice.file_path and Path(voice.file_path).exists():
                Path(voice.file_path).unlink()
                self.logger.info(f"Deleted voice file: {voice.file_path}")
            
            # Remove from voices list
            self.voices = [v for v in self.voices if not (v.name == voice_name and v.engine == "custom")]
            
            # Update persistent storage
            self._save_voice_library()
            
            self.logger.info(f"Removed custom voice: {voice_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove custom voice: {e}")
            return False
    
    def get_custom_voices(self) -> List[Voice]:
        """Get list of custom voices only"""
        return [v for v in self.voices if v.engine == "custom"]
    
    def generate_speech(self, text: str, voice_name: str, output_path: str, 
                       speed: float = 1.0) -> bool:
        """Generate speech from text"""
        try:
            voice = next((v for v in self.voices if v.name == voice_name), None)
            if not voice:
                # Default to first VibeVoice voice if available
                vibevoice_voices = [v for v in self.voices if v.engine == "vibevoice"]
                if vibevoice_voices:
                    voice = vibevoice_voices[0]
                    self.logger.info(f"Voice '{voice_name}' not found, defaulting to VibeVoice: {voice.name}")
                else:
                    voice = self.voices[0] if self.voices else None
                    if voice:
                        self.logger.info(f"Voice '{voice_name}' not found, using fallback: {voice.name}")
                if not voice:
                    return False
            
            # Prioritize VibeVoice over other engines
            if voice.engine == "vibevoice" and self.vibevoice_available:
                return self._generate_vibevoice(text, voice.name, output_path, speed)
            elif voice.engine == "custom":
                # For custom voices, try to use with VibeVoice if available
                return self._generate_custom_voice(text, voice, output_path, speed)
            elif voice.engine == "espeak" and self.espeak_available:
                return self._generate_espeak(text, voice.name, output_path, speed)
            elif self.pyttsx3_available:
                return self._generate_pyttsx3(text, output_path, speed)
            else:
                # Fallback: create silent audio file
                return self._generate_silent(text, output_path)
                
        except Exception as e:
            self.logger.error(f"Speech generation failed: {e}")
            return False
    
    def _generate_vibevoice(self, text: str, voice_name: str, 
                           output_path: str, speed: float) -> bool:
        """Generate speech using VibeVoice directly via model API"""
        try:
            self.logger.info(f"Starting VibeVoice generation with text: '{text[:50]}...', voice: {voice_name}")
            
            if not VIBEVOICE_IMPORTS_OK:
                self.logger.error("VibeVoice imports not available")
                return False

            # Resolve voice sample path from demo/voices
            voices_dir = Path("demo") / "voices"
            if not voices_dir.exists():
                self.logger.error(f"Voices directory not found: {voices_dir}")
                return False

            # Try to find a matching voice file
            voice_file = None
            candidates = list(voices_dir.glob("*.wav")) + list(voices_dir.glob("*.mp3"))
            for f in candidates:
                base = f.stem.lower()
                if voice_name.lower() in base or base in voice_name.lower():
                    voice_file = f
                    break
            if voice_file is None and candidates:
                # Fallback to first available
                voice_file = candidates[0]
                self.logger.warning(f"Voice '{voice_name}' not found, using {voice_file.name}")
            if voice_file is None:
                self.logger.error("No voice samples available")
                return False

            self.logger.info(f"Using voice file: {voice_file}")

            # Build a simple one-speaker script
            # Normalize problematic unicode punctuation to plain ASCII to avoid
            # encoding / console display issues (mojibake) on Windows.
            normalized = (
                text
                .replace("\u2019", "'")  # Right single quote
                .replace("\u2018", "'")  # Left single quote
                .replace("\u201c", '"')  # Left double quote
                .replace("\u201d", '"')  # Right double quote
            )
            script = f"Speaker 1: {normalized}"
            self.logger.info(f"Generated script: {script[:100]}...")

            # Select model path (env override supported)
            model_path = os.environ.get("VIBEVOICE_MODEL", "microsoft/VibeVoice-1.5b")

            # Load & cache model/processor once
            if self._vv_model is None or self._vv_model_path != model_path:
                # Determine initial desired device
                desired_device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else (
                    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
                )

                # Probe GPU capability; fallback to CPU if unsupported kernels likely
                device = desired_device
                if device == "cuda":
                    try:
                        major, minor = torch.cuda.get_device_capability(0)
                        sm = major * 10 + minor
                        # Heuristic: if current torch build lacks this SM (compare to list from torch.cuda.get_arch_list())
                        arch_list = getattr(torch.cuda, 'get_arch_list', lambda: [])()
                        # arch_list entries like 'sm_80'; if our sm not in list, fallback
                        if arch_list and f'sm_{sm}' not in arch_list:
                            self.logger.warning(
                                f"CUDA capability sm_{sm} unsupported by this PyTorch build; falling back to CPU." )
                            device = "cpu"
                    except Exception as probe_err:
                        self.logger.warning(f"Could not probe CUDA capability, falling back to CPU: {probe_err}")
                        device = "cpu"
                if device == "mps":
                    # Keep float32 to avoid precision issues
                    load_dtype = torch.float32
                    attn_impl = "sdpa"
                elif device == "cuda":
                    load_dtype = torch.bfloat16
                    attn_impl = "flash_attention_2"
                else:
                    load_dtype = torch.float32
                    attn_impl = "sdpa"

                self.logger.info(f"Loading VibeVoice model on device={device} dtype={load_dtype}")
                processor = VibeVoiceProcessor.from_pretrained(model_path)
                try:
                    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        model_path,
                        torch_dtype=load_dtype,
                        device_map=(device if device in ("cuda", "cpu") else None),
                        attn_implementation=attn_impl,
                    )
                    if device == "mps":
                        model.to("mps")
                except Exception as load_err:
                    if device == "cuda" and "flash_attn" in str(load_err):
                        self.logger.warning(f"Flash Attention 2 failed, retrying with SDPA: {load_err}")
                        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            model_path,
                            torch_dtype=load_dtype,
                            device_map=(device if device in ("cuda", "cpu") else None),
                            attn_implementation="sdpa",
                        )
                    else:
                        self.logger.warning(f"Primary load failed ({load_err}); retrying with SDPA + float32 on CPU")
                        device = "cpu"
                        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            model_path,
                            torch_dtype=torch.float32,
                            device_map="cpu",
                            attn_implementation="sdpa",
                        )

                model.eval()
                if hasattr(model, "set_ddpm_inference_steps"):
                    model.set_ddpm_inference_steps(num_steps=8)

                # Cache
                self._vv_model = model
                self._vv_processor = processor
                self._vv_device = device
                self._vv_model_path = model_path
            else:
                model = self._vv_model
                processor = self._vv_processor
                device = self._vv_device
                
            self.logger.info(f"Using model: {type(model)}, processor: {type(processor)}, device: {device}")
            self.logger.info(f"Model is None: {model is None}, Processor is None: {processor is None}")
            
            # Debug the variables we're about to use
            self.logger.info(f"Script to process: {script}")
            self.logger.info(f"Voice file path: {voice_file}")
            self.logger.info(f"Voice file exists: {voice_file.exists() if voice_file else 'None'}")

            # Prepare inputs
            self.logger.info("Processing inputs...")
            inputs = processor(
                text=[script],
                voice_samples=[[str(voice_file)]],  # list of list for batch & speakers
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Move inputs to device with error handling
            self.logger.info(f"Moving inputs to device: {device}")
            for k, v in inputs.items():
                if hasattr(torch, "is_tensor") and torch.is_tensor(v):
                    if v is not None:
                        try:
                            inputs[k] = v.to(device)
                            self.logger.info(f"Moved {k} (shape: {v.shape}) to {device}")
                        except Exception as move_err:
                            self.logger.warning(f"Failed to move {k} to {device}: {move_err}")
                            if device != "cpu":
                                self.logger.warning("Falling back to CPU")
                                device = "cpu"
                                inputs[k] = v.cpu()
                    else:
                        self.logger.warning(f"Input {k} is None!")
                else:
                    self.logger.info(f"Skipping non-tensor input: {k}")

            # Generate
            try:
                self.logger.info(f"Starting generation on device: {device}")
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=max(0.8, min(2.0, 1.3 / max(0.5, min(2.0, speed)))),
                    tokenizer=processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                )
                self.logger.info(f"Generation completed. Outputs type: {type(outputs)}")
                if hasattr(outputs, 'speech_outputs'):
                    self.logger.info(f"Speech outputs length: {len(outputs.speech_outputs) if outputs.speech_outputs else 'None'}")
                else:
                    self.logger.warning("Outputs has no speech_outputs attribute")
            except Exception as gen_err:
                if self._vv_device == "cuda":
                    self.logger.warning(f"Generation on CUDA failed ({gen_err}); switching cached model to CPU and retrying once.")
                    # Move to CPU & retry
                    try:
                        model.to("cpu")
                        self._vv_device = "cpu"
                        for k, v in inputs.items():
                            if hasattr(torch, "is_tensor") and torch.is_tensor(v):
                                inputs[k] = v.cpu()
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=None,
                            cfg_scale=max(0.8, min(2.0, 1.3 / max(0.5, min(2.0, speed)))),
                            tokenizer=processor.tokenizer,
                            generation_config={"do_sample": False},
                            verbose=False,
                        )
                    except Exception as retry_err:
                        self.logger.error(f"Retry on CPU failed: {retry_err}")
                        return False
                else:
                    self.logger.error(f"VibeVoice generation failed: {gen_err}")
                    return False

            # Save audio
            try:
                if outputs and hasattr(outputs, 'speech_outputs') and outputs.speech_outputs:
                    self.logger.info("Saving generated audio")
                    processor.save_audio(
                        outputs.speech_outputs[0],
                        output_path=output_path,
                    )
                else:
                    self.logger.error("No speech outputs to save")
                    return False
            except Exception as save_err:
                self.logger.error(f"Failed to save audio: {save_err}")
                return False

            return Path(output_path).exists()

        except Exception as e:
            self.logger.error(f"VibeVoice generation failed: {e}")
            return False
    
    def _generate_espeak(self, text: str, voice_name: str, 
                        output_path: str, speed: float) -> bool:
        """Generate speech using eSpeak"""
        try:
            # Calculate speed (eSpeak uses words per minute)
            wpm = int(180 * speed)  # Base 180 WPM
            
            cmd = [
                "espeak",
                "-v", voice_name,
                "-s", str(wpm),
                "-w", output_path,
                text
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0 and Path(output_path).exists()
            
        except Exception as e:
            self.logger.error(f"eSpeak generation failed: {e}")
            return False
    
    def _generate_pyttsx3(self, text: str, output_path: str, speed: float) -> bool:
        """Generate speech using pyttsx3"""
        try:
            if not PYTTSX3_AVAILABLE or pyttsx3 is None:
                self.logger.error("pyttsx3 not available")
                return False
            
            engine = pyttsx3.init()
            engine.setProperty('rate', int(200 * speed))
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            
            return Path(output_path).exists()
            
        except Exception as e:
            self.logger.error(f"pyttsx3 generation failed: {e}")
            return False
    
    def _generate_custom_voice(self, text: str, voice: Voice, 
                              output_path: str, speed: float) -> bool:
        """Generate speech using a custom voice file"""
        try:
            if not voice.file_path or not Path(voice.file_path).exists():
                self.logger.error(f"Custom voice file not found: {voice.file_path}")
                return False
            
            # Use VibeVoice with custom voice sample if available
            if self.vibevoice_available and VIBEVOICE_IMPORTS_OK:
                return self._generate_vibevoice_with_custom(text, voice.file_path, output_path, speed)
            else:
                # Fallback: copy the custom voice file or create silent audio
                self.logger.warning(f"VibeVoice not available for custom voice {voice.name}, using fallback")
                return self._generate_silent(text, output_path)
                
        except Exception as e:
            self.logger.error(f"Custom voice generation failed: {e}")
            return False
    
    def _generate_vibevoice_with_custom(self, text: str, voice_file_path: str,
                                       output_path: str, speed: float) -> bool:
        """Generate speech using VibeVoice with a custom voice sample"""
        try:
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_path = "VibeVoice/VibeVoice"
            
            # Load model and processor
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path, device_map=device, torch_dtype=torch.float16
            )
            processor = VibeVoiceProcessor.from_pretrained(model_path, device=device)
            
            # Process custom voice sample
            inputs = processor(
                text_list=[text],
                voice_sample_list=[voice_file_path],
                device=device,
                pad_audio_to_same_length=True,
            )
            
            # Move inputs to device
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(device)
            
            # Generate with speed adjustment
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=max(0.8, min(2.0, 1.3 / max(0.5, min(2.0, speed)))),
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
            )
            
            # Save audio
            processor.save_audio(
                outputs.speech_outputs[0],
                output_path=output_path,
            )
            
            return Path(output_path).exists()
            
        except Exception as e:
            self.logger.error(f"VibeVoice custom voice generation failed: {e}")
            return False
    
    def _generate_silent(self, text: str, output_path: str) -> bool:
        """Generate silent audio as fallback"""
        try:
            if not AUDIO_AVAILABLE:
                # Create empty file
                Path(output_path).touch()
                return True
            
            # Create silent audio based on text length
            word_count = len(text.split())
            duration = word_count * 0.4  # Rough estimate: 0.4 seconds per word
            sample_rate = 22050
            samples = int(duration * sample_rate)
            
            # Generate silent audio
            audio_data = np.zeros(samples, dtype=np.float32)
            sf.write(output_path, audio_data, sample_rate)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Silent audio generation failed: {e}")
            return False

# For backward compatibility
class MultiModelTTSBackend(SimpleTTSBackend):
    """Alias for backward compatibility"""
    
    def __init__(self):
        super().__init__()
        self.vibevoice_available = self.vibevoice_available
        self.coqui_available = False  # Not available in simplified version
    
    def discover_voices(self) -> List[Voice]:
        """Alias for get_voices"""
        return self.get_voices()

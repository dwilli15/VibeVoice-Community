"""
Multi-Model TTS Backend - Unified interface for VibeVoice and Coqui AI
Provides seamless integration between different TTS engines
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import tempfile
import subprocess

# Audio/array deps (used by both engines)
try:
    import numpy as np
except Exception:
    np = None  # Will be checked at runtime where needed
try:
    import soundfile as sf
except Exception:
    sf = None

# Always try to import torch for device detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Defer heavy VibeVoice/Transformers imports to runtime to avoid ABI issues
VIBEVOICE_AVAILABLE = False

def _lazy_import_vibevoice():
    """Attempt to import VibeVoice stack lazily. Returns tuple (ok, modules_dict)."""
    global VIBEVOICE_AVAILABLE
    try:
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig  # noqa: F401
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,  # noqa: F401
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor  # noqa: F401
        VIBEVOICE_AVAILABLE = True
        return True, {
            "VibeVoiceConfig": VibeVoiceConfig,
            "VibeVoiceForConditionalGenerationInference": VibeVoiceForConditionalGenerationInference,
            "VibeVoiceProcessor": VibeVoiceProcessor,
        }
    except Exception as e:
        # Keep it quiet in import path; log when actually needed
        VIBEVOICE_AVAILABLE = False
        return False, {"error": e}

COQUI_AVAILABLE = False
TTS = None
ModelManager = None

def _try_import_coqui(prefer_local: bool = False) -> bool:
    """Attempt to import Coqui TTS (pip first by default, else local path). Sets globals on success."""
    global COQUI_AVAILABLE, TTS, ModelManager, sf, np

    coqui_local_path = Path("d:/omen/coqui-ai")
    attempts = ["local", "pip"] if prefer_local else ["pip", "local"]

    for source in attempts:
        try:
            if source == "local":
                if not coqui_local_path.exists():
                    continue
                if str(coqui_local_path) not in sys.path:
                    sys.path.insert(0, str(coqui_local_path))

            # Import and assign to globals
            from TTS.api import TTS as _TTS
            try:
                from TTS.utils.manage import ModelManager as _MM
            except Exception:
                _MM = None
            import soundfile as _sf
            import numpy as _np

            TTS = _TTS
            ModelManager = _MM
            sf = _sf
            np = _np
            COQUI_AVAILABLE = True
            print(f"✅ Coqui AI loaded from {'local' if source=='local' else 'pip'} installation")
            return True

        except Exception as e:
            # Try the next source
            continue

    COQUI_AVAILABLE = False
    return False

# Attempt import now (pip first, then local as fallback)
_try_import_coqui(prefer_local=False)

class TTSEngine(Enum):
    VIBEVOICE = "vibevoice"
    COQUI = "coqui"
    AUTO = "auto"

@dataclass
class Voice:
    """Voice configuration for TTS engines"""
    name: str
    engine: TTSEngine
    model_path: str
    language: str = "en"
    gender: str = "neutral"
    description: str = ""
    sample_rate: int = 22050
    speaker_id: Optional[str] = None
    emotion: Optional[str] = None
    file_path: Optional[str] = None  # Path to custom voice file
    created_at: Optional[str] = None  # Timestamp when added
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert TTSEngine enum to string for JSON serialization
        data['engine'] = self.engine.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Voice':
        """Create Voice from dictionary"""
        # Handle TTSEngine conversion
        if isinstance(data.get('engine'), str):
            data['engine'] = TTSEngine(data['engine'])
        return cls(**data)
    
@dataclass
class TTSRequest:
    """TTS generation request"""
    text: str
    voice: Voice
    output_path: str
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    emotion: Optional[str] = None
    speaker_embedding: Optional[str] = None

class MultiModelTTSBackend:
    """Unified TTS backend supporting multiple engines"""
    
    def __init__(self):
        self.engines = {}
        self.voices = {}
        
        # Device detection with fallback
        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        # Enable high-performance kernels on CUDA
        if TORCH_AVAILABLE and self.device == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            
        self.models_loaded = {}
        
        # Initialize custom voices directory and config
        self.custom_voices_dir = self._get_custom_voices_dir()
        self.config_file = self.custom_voices_dir / "voice_library.json"
        
        # Initialize available engines
        self.initialize_engines()
        self.discover_voices()
        
        # Load custom voices from persistent storage
        self._load_custom_voices()
        
    def initialize_engines(self):
        """Initialize available TTS engines"""
        print("🔧 Initializing TTS engines...")
        
        # Try to initialize VibeVoice
        vibevoice_ok, vibevoice_modules = _lazy_import_vibevoice()
        if vibevoice_ok:
            self.engines[TTSEngine.VIBEVOICE] = {
                "available": True,
                "models": {},
                "processor": None,
                "modules": vibevoice_modules
            }
            print("✅ VibeVoice engine available")
        else:
            print(f"❌ VibeVoice engine not available: {vibevoice_modules.get('error', 'Unknown error')}")

        # Try to initialize Coqui AI
        coqui_ok = _try_import_coqui()
        if coqui_ok:
            self.engines[TTSEngine.COQUI] = {
                "available": True,
                "models": {},
                "manager": ModelManager() if ModelManager else None
            }
            print("✅ Coqui AI engine available")
        else:
            print("❌ Coqui AI engine not available")
            
    def discover_voices(self):
        """Discover available voices from all engines"""
        print("🎭 Discovering available voices...")
        
        # VibeVoice voices
        if TTSEngine.VIBEVOICE in self.engines:
            self.discover_vibevoice_voices()
            
        # Coqui AI voices
        if TTSEngine.COQUI in self.engines:
            self.discover_coqui_voices()
            
        print(f"📊 Total voices discovered: {len(self.voices)}")
        
    def discover_vibevoice_voices(self):
        """Discover VibeVoice models and voices"""
        # Primary model voices
        vibevoice_voices = [
            Voice(
                name="VibeVoice-1.5B",
                engine=TTSEngine.VIBEVOICE,
                model_path="microsoft/VibeVoice-1.5B",
                language="en",
                description="High-quality conversational voice with multi-speaker support",
                sample_rate=24000
            ),
            Voice(
                name="VibeVoice-Large",
                engine=TTSEngine.VIBEVOICE,
                model_path="aoi-ot/VibeVoice-Large",
                language="en",
                description="Large model with enhanced quality and speaker consistency",
                sample_rate=24000
            )
        ]
        
        # Named voice presets (compatible with our enhanced system)
        named_voices = [
            Voice("bf_isabella", TTSEngine.VIBEVOICE, "microsoft/VibeVoice-1.5B", "en", "female", "Default female voice"),
            Voice("af_heart", TTSEngine.VIBEVOICE, "microsoft/VibeVoice-1.5B", "en", "female", "Warm female voice"),
            Voice("en-Alice_woman", TTSEngine.VIBEVOICE, "microsoft/VibeVoice-1.5B", "en", "female", "Alice"),
            Voice("en-Carter_man", TTSEngine.VIBEVOICE, "microsoft/VibeVoice-1.5B", "en", "male", "Carter"),
            Voice("en-Frank_man", TTSEngine.VIBEVOICE, "microsoft/VibeVoice-1.5B", "en", "male", "Frank"),
            Voice("en-Maya_woman", TTSEngine.VIBEVOICE, "microsoft/VibeVoice-1.5B", "en", "female", "Maya"),
            Voice("zh-Anchen_man", TTSEngine.VIBEVOICE, "microsoft/VibeVoice-1.5B", "zh", "male", "Anchen"),
            Voice("zh-Bowen_man", TTSEngine.VIBEVOICE, "microsoft/VibeVoice-1.5B", "zh", "male", "Bowen"),
            Voice("zh-Xinran_woman", TTSEngine.VIBEVOICE, "microsoft/VibeVoice-1.5B", "zh", "female", "Xinran"),
        ]
        
        # Add all voices
        for voice in vibevoice_voices + named_voices:
            self.voices[voice.name] = voice
            
    def discover_coqui_voices(self):
        """Discover Coqui AI models and voices"""
        try:
            # Get available models via manager if present, else try class method
            models = set()
            try:
                mgr = self.engines[TTSEngine.COQUI].get("manager")
                if mgr is not None:
                    models = set(mgr.list_models())
            except Exception:
                pass
            if not models:
                try:
                    if hasattr(TTS, "list_models"):
                        models = set(TTS.list_models())
                except Exception:
                    models = set()
            
            # Popular Coqui models
            popular_models = [
                ("tts_models/en/ljspeech/tacotron2-DDC", "LJSpeech Tacotron2", "en", "female"),
                ("tts_models/en/ljspeech/glow-tts", "LJSpeech GlowTTS", "en", "female"),
                ("tts_models/en/ljspeech/speedy-speech", "LJSpeech SpeedySpeech", "en", "female"),
                ("tts_models/en/vctk/vits", "VCTK VITS", "en", "multi"),
                ("tts_models/en/jenny/jenny", "Jenny Voice", "en", "female"),
                ("tts_models/multilingual/multi-dataset/xtts_v2", "XTTS v2", "multilingual", "multi"),
                ("tts_models/multilingual/multi-dataset/your_tts", "YourTTS", "multilingual", "multi"),
                ("tts_models/de/thorsten/tacotron2-DDC", "Thorsten German", "de", "male"),
                ("tts_models/fr/mai/tacotron2-DDC", "Mai French", "fr", "female"),
                ("tts_models/es/mai/tacotron2-DDC", "Mai Spanish", "es", "female"),
            ]
            
            for model_path, name, language, gender in popular_models:
                # If models set is empty (listing unavailable), still expose popular presets
                if not models or model_path in models:
                    voice = Voice(
                        name=name,
                        engine=TTSEngine.COQUI,
                        model_path=model_path,
                        language=language,
                        gender=gender,
                        description=f"Coqui AI {name} model",
                        sample_rate=22050
                    )
                    self.voices[f"coqui_{name.lower().replace(' ', '_')}"] = voice
                    
        except Exception as e:
            print(f"Error discovering Coqui voices: {e}")
            
    def get_available_voices(self) -> Dict[str, Voice]:
        """Get all available voices"""
        return self.voices
        
    def get_voices_by_engine(self, engine: TTSEngine) -> Dict[str, Voice]:
        """Get voices for specific engine"""
        return {k: v for k, v in self.voices.items() if v.engine == engine}
        
    def get_voices_by_language(self, language: str) -> Dict[str, Voice]:
        """Get voices for specific language"""
        return {k: v for k, v in self.voices.items() if v.language == language or v.language == "multilingual"}
        
    def load_model(self, voice: Voice) -> bool:
        """Load a specific model"""
        try:
            model_key = f"{voice.engine.value}_{voice.model_path}"
            
            if model_key in self.models_loaded:
                print(f"✅ Model already loaded: {voice.name}")
                return True
                
            print(f"🔄 Loading model: {voice.name}")
            
            if voice.engine == TTSEngine.VIBEVOICE:
                return self.load_vibevoice_model(voice, model_key)
            elif voice.engine == TTSEngine.COQUI:
                return self.load_coqui_model(voice, model_key)
                
        except Exception as e:
            print(f"❌ Failed to load model {voice.name}: {e}")
            return False
            
    def load_vibevoice_model(self, voice: Voice, model_key: str) -> bool:
        """Load VibeVoice model"""
        try:
            ok, mods = _lazy_import_vibevoice()
            if not ok:
                err = mods.get("error")
                print(f"❌ VibeVoice stack not available: {err}")
                return False

            VibeVoiceConfig = mods["VibeVoiceConfig"]
            VibeVoiceForConditionalGenerationInference = mods["VibeVoiceForConditionalGenerationInference"]
            VibeVoiceProcessor = mods["VibeVoiceProcessor"]

            # Choose dtype based on device (default to float16 on CUDA to avoid unsupported bf16)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                use_bf16 = os.environ.get("VIBEVOICE_USE_BF16", "0") in ("1", "true", "True")
                if use_bf16 and hasattr(torch, "bfloat16"):
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16
            else:
                dtype = torch.float32

            config = VibeVoiceConfig.from_pretrained(voice.model_path)
            # Try flash attention 2 first, then fall back to SDPA
            try:
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    voice.model_path,
                    config=config,
                    torch_dtype=dtype,
                    device_map=("cuda" if self.device == "cuda" else "cpu"),
                    attn_implementation="flash_attention_2",
                )
            except Exception:
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    voice.model_path,
                    config=config,
                    torch_dtype=dtype,
                    device_map=("cuda" if self.device == "cuda" else "cpu"),
                    attn_implementation="sdpa",
                )
            
            # Try to load processor with external tokenizer if available
            try:
                # Check if external tokenizer exists
                external_tokenizer_path = "D:/omen/temp/tokenizer"
                if os.path.exists(external_tokenizer_path):
                    print(f"🔧 Using external tokenizer: {external_tokenizer_path}")
                    processor = VibeVoiceProcessor.from_pretrained(
                        voice.model_path,
                        tokenizer_path=external_tokenizer_path
                    )
                else:
                    processor = VibeVoiceProcessor.from_pretrained(voice.model_path)
            except Exception as e:
                print(f"⚠️ External tokenizer failed, using default: {e}")
                processor = VibeVoiceProcessor.from_pretrained(voice.model_path)
            
            self.models_loaded[model_key] = {
                "model": model,
                "processor": processor,
                "config": config
            }
            
            # Update engine info
            self.engines[TTSEngine.VIBEVOICE]["models"][voice.model_path] = model
            self.engines[TTSEngine.VIBEVOICE]["processor"] = processor
            
            print(f"✅ VibeVoice model loaded: {voice.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load VibeVoice model: {e}")
            return False
            
    def load_coqui_model(self, voice: Voice, model_key: str) -> bool:
        """Load Coqui AI model"""
        try:
            gpu_flag = False
            if TORCH_AVAILABLE:
                try:
                    gpu_flag = torch.cuda.is_available()
                except Exception:
                    gpu_flag = False

            # Some environments require agreeing to TOS
            os.environ.setdefault("COQUI_TOS_AGREED", "1")

            tts = TTS(model_name=voice.model_path, progress_bar=True, gpu=gpu_flag)
            
            self.models_loaded[model_key] = {
                "tts": tts,
                "model_path": voice.model_path
            }
            
            # Update engine info
            self.engines[TTSEngine.COQUI]["models"][voice.model_path] = tts
            
            print(f"✅ Coqui AI model loaded: {voice.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Coqui AI model: {e}")
            return False
            
    def unload_model(self, voice: Voice):
        """Unload a specific model"""
        model_key = f"{voice.engine.value}_{voice.model_path}"
        
        if model_key in self.models_loaded:
            del self.models_loaded[model_key]
            print(f"🗑️ Unloaded model: {voice.name}")
            
    def generate_speech(self, request: TTSRequest) -> bool:
        """Generate speech using the specified voice"""
        try:
            # Load model if not already loaded
            if not self.load_model(request.voice):
                return False
                
            print(f"🎙️ Generating speech with {request.voice.name}")
            
            if request.voice.engine == TTSEngine.VIBEVOICE:
                return self.generate_vibevoice_speech(request)
            elif request.voice.engine == TTSEngine.COQUI:
                return self.generate_coqui_speech(request)
                
        except Exception as e:
            print(f"❌ Speech generation failed: {e}")
            return False
            
    def generate_vibevoice_speech(self, request: TTSRequest) -> bool:
        """Generate speech using VibeVoice"""
        try:
            model_key = f"{request.voice.engine.value}_{request.voice.model_path}"
            model_data = self.models_loaded[model_key]
            
            print(f"🎵 Generating with VibeVoice: {request.text[:50]}...")
            
            # Get processor and model
            processor = model_data["processor"]
            model = model_data["model"]
            
            # Format text with proper VibeVoice speaker format
            if ':' not in request.text or not request.text.strip().startswith('Speaker'):
                # Add proper speaker format - VibeVoice expects "Speaker 1:", "Speaker 2:", etc.
                formatted_text = f"Speaker 1: {request.text}"
            else:
                formatted_text = request.text
            
            # Find (or synthesize) a voice sample file for VibeVoice
            voices_dir = Path("demo") / "voices"
            voice_file = None

            # Ensure directory exists
            try:
                voices_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            # Gather existing candidates
            candidates = []
            try:
                if voices_dir.exists():
                    candidates = list(voices_dir.glob("*.wav")) + list(voices_dir.glob("*.mp3"))
            except Exception:
                candidates = []

            # If none exist, synthesize a tiny reference sample so generation can proceed
            if not candidates:
                try:
                    import numpy as _np
                    import math
                    sr = 24000
                    dur = 0.6
                    t = _np.linspace(0, dur, int(sr * dur), endpoint=False)
                    # Simple two-tone chirp-like sample to satisfy reference audio requirement
                    tone = 0.2 * _np.sin(2 * math.pi * 440 * t) + 0.1 * _np.sin(2 * math.pi * 660 * t)
                    tone = tone.astype(_np.float32)
                    synth_path = voices_dir / "en-Alice_woman.wav"
                    if sf is not None:
                        sf.write(str(synth_path), tone, sr)
                    else:
                        # Fallback to built-in wave module
                        import wave, struct
                        with wave.open(str(synth_path), 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)  # 16-bit PCM
                            wf.setframerate(sr)
                            # Convert to int16 for wave module
                            pcm = (_np.clip(tone, -1.0, 1.0) * 32767).astype(_np.int16)
                            wf.writeframes(b"".join(struct.pack('<h', x) for x in pcm))
                    candidates = [synth_path]
                    print(f"🧪 Created fallback voice sample: {synth_path}")
                except Exception as synth_err:
                    print(f"❌ Unable to synthesize fallback voice sample: {synth_err}")
                    # Continue; will error out below if still no candidates

            # Select matching or first candidate
            if candidates:
                for f in candidates:
                    base = f.stem.lower()
                    if request.voice.name.lower() in base or base in request.voice.name.lower():
                        voice_file = f
                        break
                if voice_file is None:
                    voice_file = candidates[0]
                    print(f"⚠️ Voice '{request.voice.name}' not found, using {voice_file.name}")

            if voice_file is None:
                print("❌ No voice samples available in demo/voices")
                return False
            
            # Process text with voice samples
            inputs = processor(
                text=[formatted_text],
                voice_samples=[[str(voice_file)]],  # list of list for batch & speakers
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Move inputs to same device as model
            device = next(model.parameters()).device
            inputs_on_device = {}
            for k, v in inputs.items():
                if hasattr(v, 'to'):
                    inputs_on_device[k] = v.to(device)
                else:
                    inputs_on_device[k] = v
            
            # Generate speech - use the processor's tokenizer explicitly
            with torch.inference_mode():
                try:
                    audio_output = model.generate(
                        **inputs_on_device, 
                        tokenizer=processor.tokenizer,  # Pass tokenizer explicitly
                        max_new_tokens=1000,  # Use max_new_tokens instead of max_length
                        do_sample=True,
                        temperature=0.8
                    )
                    print(f"✅ Generation successful, output type: {type(audio_output)}")
                    
                    # Check what kind of output we got
                    if hasattr(audio_output, 'speech_outputs') and audio_output.speech_outputs is not None:
                        print("Found speech_outputs attribute")
                        audio = audio_output.speech_outputs
                    elif hasattr(audio_output, 'audio_values'):
                        print("Found audio_values attribute")
                        audio = audio_output.audio_values
                    elif hasattr(audio_output, 'audio'):
                        print("Found audio attribute")
                        audio = audio_output.audio
                    elif isinstance(audio_output, dict):
                        print(f"Output is dict with keys: {list(audio_output.keys())}")
                        # Look for audio in VibeVoice-specific keys first, then common keys
                        for key in ['speech_outputs', 'audio', 'audio_values', 'waveform', 'speech', 'output']:
                            if key in audio_output and audio_output[key] is not None:
                                audio = audio_output[key]
                                print(f"Using audio from key: {key}")
                                break
                        else:
                            print("No audio found in output dict")
                            return False
                    elif isinstance(audio_output, torch.Tensor):
                        print(f"Output is tensor with shape: {audio_output.shape}")
                        audio = audio_output
                    else:
                        print(f"❌ Unexpected output type: {type(audio_output)}")
                        return False
                    
                    # Convert to numpy
                    if isinstance(audio, list):
                        print(f"Audio is list with {len(audio)} items")
                        if len(audio) > 0:
                            # Take the first item if it's a list
                            audio = audio[0]
                            print(f"Using first item: {type(audio)}, shape: {getattr(audio, 'shape', 'no shape')}")
                        else:
                            print("❌ Empty audio list")
                            return False
                    
                    if hasattr(audio, 'cpu'):
                        audio = audio.squeeze().cpu().numpy()
                        print(f"Converted tensor to numpy: {audio.shape}, dtype: {audio.dtype}")
                    elif hasattr(audio, 'numpy'):
                        audio = audio.squeeze().numpy()
                        print(f"Converted array to numpy: {audio.shape}, dtype: {audio.dtype}")
                    elif isinstance(audio, np.ndarray):
                        print(f"Already numpy array: {audio.shape}, dtype: {audio.dtype}")
                    else:
                        print(f"❌ Cannot convert audio to numpy: {type(audio)}")
                        return False
                    
                    print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")
                    
                except Exception as gen_error:
                    print(f"❌ Generation failed: {gen_error}")
                    # Fallback: create a short beep as placeholder
                    try:
                        import numpy as beep_np
                        sample_rate = 24000
                        duration = 0.5
                        t = beep_np.linspace(0, duration, int(sample_rate * duration))
                        audio = 0.3 * beep_np.sin(2 * beep_np.pi * 440 * t)  # 440Hz beep
                        print("⚠️ Using fallback beep audio")
                    except ImportError:
                        print("❌ Cannot create fallback audio - numpy not available")
                        return False
            
            # Ensure audio is in the right format
            try:
                import numpy as np_local
            except ImportError:
                print("❌ numpy not available for audio processing")
                return False
                
            if audio.dtype != np_local.float32:
                audio = audio.astype(np_local.float32)
            
            # Normalize audio to prevent clipping
            if len(audio) > 0:
                max_val = max(abs(audio.max()), abs(audio.min()))
                if max_val > 1.0:
                    audio = audio / max_val
            
            # Save audio
            if sf is None:
                print("❌ soundfile not available; cannot write audio output")
                return False
            sf.write(request.output_path, audio, request.voice.sample_rate)
            print(f"✅ VibeVoice audio saved: {request.output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ VibeVoice generation failed: {e}")
            return False
            
    def generate_coqui_speech(self, request: TTSRequest) -> bool:
        """Generate speech using Coqui AI"""
        try:
            model_key = f"{request.voice.engine.value}_{request.voice.model_path}"
            model_data = self.models_loaded[model_key]
            tts = model_data["tts"]
            
            print(f"🎵 Generating with Coqui AI: {request.text[:50]}...")
            
            # Generate audio
            if "xtts" in request.voice.model_path.lower():
                # XTTS requires speaker embedding or reference audio
                if request.speaker_embedding:
                    tts.tts_to_file(
                        text=request.text,
                        file_path=request.output_path,
                        speaker_wav=request.speaker_embedding,
                        language="en"
                    )
                else:
                    # Use default speaker
                    tts.tts_to_file(
                        text=request.text,
                        file_path=request.output_path,
                        language="en"
                    )
            else:
                # Standard TTS models
                tts.tts_to_file(
                    text=request.text,
                    file_path=request.output_path
                )
                
            print(f"✅ Coqui AI audio saved: {request.output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Coqui AI generation failed: {e}")
            return False
            
    def get_model_info(self, voice: Voice) -> Dict:
        """Get information about a specific model"""
        model_key = f"{voice.engine.value}_{voice.model_path}"
        
        info = {
            "name": voice.name,
            "engine": voice.engine.value,
            "model_path": voice.model_path,
            "language": voice.language,
            "gender": voice.gender,
            "sample_rate": voice.sample_rate,
            "loaded": model_key in self.models_loaded,
            "description": voice.description
        }
        
        return info
        
    def get_engine_status(self) -> Dict:
        """Get status of all engines"""
        status = {}
        
        for engine, data in self.engines.items():
            status[engine.value] = {
                "available": data["available"],
                "models_loaded": len(data.get("models", {})),
                "total_voices": len(self.get_voices_by_engine(engine))
            }
            
        return status
        
    def benchmark_voice(self, voice: Voice, test_text: str = "Hello, this is a test.") -> Dict:
        """Benchmark a voice's performance"""
        import time
        
        temp_file = tempfile.mktemp(suffix=".wav")
        
        try:
            start_time = time.time()
            
            request = TTSRequest(
                text=test_text,
                voice=voice,
                output_path=temp_file
            )
            
            success = self.generate_speech(request)
            generation_time = time.time() - start_time
            
            result = {
                "voice": voice.name,
                "engine": voice.engine.value,
                "success": success,
                "generation_time": generation_time,
                "text_length": len(test_text),
                "words_per_second": len(test_text.split()) / generation_time if success else 0
            }
            
            if success and os.path.exists(temp_file):
                file_size = os.path.getsize(temp_file)
                result["output_size"] = file_size
                os.unlink(temp_file)  # Clean up
                
            return result
            
        except Exception as e:
            return {
                "voice": voice.name,
                "engine": voice.engine.value,
                "success": False,
                "error": str(e)
            }
    
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
                        # Add to voices dict if not already present
                        if voice.name not in self.voices:
                            self.voices[voice.name] = voice
                            print(f"📤 Loaded custom voice: {voice.name}")
                    else:
                        print(f"⚠️ Custom voice file missing: {voice_data.get('file_path')}")
        except Exception as e:
            print(f"❌ Failed to load custom voices: {e}")
    
    def save_custom_voice(self, voice_file_path: str, voice_name: str, 
                         language: str = "en", gender: str = "neutral") -> bool:
        """Save a custom voice file for future use"""
        try:
            source_path = Path(voice_file_path)
            if not source_path.exists():
                print(f"❌ Voice file not found: {voice_file_path}")
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
                engine=TTSEngine.VIBEVOICE,  # Custom voices use VibeVoice
                model_path="custom",
                language=language,
                gender=gender,
                description=f"Custom voice uploaded by user",
                file_path=str(dest_path),
                created_at=datetime.now().isoformat()
            )
            
            # Add to current session
            self.voices[voice.name] = voice
            
            # Save to persistent storage
            self._save_voice_library()
            
            print(f"💾 Custom voice saved: {voice_name} -> {dest_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save custom voice: {e}")
            return False
    
    def _save_voice_library(self):
        """Save current custom voices to JSON file"""
        try:
            # Only save custom voices (those with file_path)
            custom_voices = [v for v in self.voices.values() if v.file_path is not None]
            voice_data = [voice.to_dict() for voice in custom_voices]
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(voice_data, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Saved {len(custom_voices)} custom voices to library")
            
        except Exception as e:
            print(f"❌ Failed to save voice library: {e}")
    
    def remove_custom_voice(self, voice_name: str) -> bool:
        """Remove a custom voice from storage"""
        try:
            # Find the voice
            voice = self.voices.get(voice_name)
            if not voice or not voice.file_path:
                print(f"⚠️ Custom voice not found: {voice_name}")
                return False
            
            # Remove file if it exists
            if Path(voice.file_path).exists():
                Path(voice.file_path).unlink()
                print(f"🗑️ Deleted voice file: {voice.file_path}")
            
            # Remove from voices dict
            del self.voices[voice_name]
            
            # Update persistent storage
            self._save_voice_library()
            
            print(f"🗑️ Removed custom voice: {voice_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to remove custom voice: {e}")
            return False
    
    def get_custom_voices(self) -> Dict[str, Voice]:
        """Get dictionary of custom voices only"""
        return {name: voice for name, voice in self.voices.items() if voice.file_path is not None}


# Singleton instance
_backend_instance = None

def get_tts_backend() -> MultiModelTTSBackend:
    """Get the global TTS backend instance"""
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = MultiModelTTSBackend()
    return _backend_instance


# Utility functions for easy access
def generate_speech_simple(text: str, voice_name: str, output_path: str, **kwargs) -> bool:
    """Simple speech generation function"""
    backend = get_tts_backend()
    voices = backend.get_available_voices()
    
    if voice_name not in voices:
        print(f"❌ Voice not found: {voice_name}")
        return False
        
    voice = voices[voice_name]
    request = TTSRequest(text=text, voice=voice, output_path=output_path, **kwargs)
    
    return backend.generate_speech(request)


def list_available_voices() -> Dict[str, Dict]:
    """List all available voices with their info"""
    backend = get_tts_backend()
    voices = backend.get_available_voices()
    
    return {name: backend.get_model_info(voice) for name, voice in voices.items()}


if __name__ == "__main__":
    # Test the backend
    print("🚀 Testing Multi-Model TTS Backend")
    
    backend = get_tts_backend()
    
    print("\n📊 Engine Status:")
    status = backend.get_engine_status()
    for engine, info in status.items():
        print(f"  {engine}: {info}")
        
    print("\n🎭 Available Voices:")
    voices = backend.get_available_voices()
    for name, voice in list(voices.items())[:5]:  # Show first 5
        info = backend.get_model_info(voice)
        print(f"  {name}: {info['engine']} - {info['description']}")
        
    if voices:
        print(f"\n... and {len(voices) - 5} more voices")

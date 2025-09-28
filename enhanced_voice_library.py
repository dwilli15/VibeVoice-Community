"""
Enhanced Voice Library Manager for VibeVoice Community
Manages and expands voice options across multiple TTS engines
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

@dataclass
class VoiceInfo:
    """Comprehensive voice information"""
    id: str
    name: str
    display_name: str
    language: str
    language_code: str
    country: str
    gender: str
    age: str
    style: str
    quality: str
    engine: str
    model_path: str
    description: str
    sample_url: Optional[str] = None
    preview_text: str = "Hello, this is a voice preview."
    is_premium: bool = False
    is_clone: bool = False
    tags: List[str] = None
    accent: str = "neutral"
    speed_range: Tuple[float, float] = (0.5, 2.0)
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class VoiceCategory(Enum):
    """Voice categories for organization"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    NARRATIVE = "narrative"
    CHARACTER = "character"
    EDUCATIONAL = "educational"
    NEWS = "news"
    STORYTELLING = "storytelling"
    CONVERSATIONAL = "conversational"

class VoiceEngine(Enum):
    """Supported TTS engines"""
    VIBEVOICE = "vibevoice"
    COQUI = "coqui"
    SIMPLE = "simple"
    CUSTOM = "custom"

class EnhancedVoiceLibrary:
    """Enhanced voice discovery and management system"""
    
    def __init__(self):
        self.voices: Dict[str, VoiceInfo] = {}
        self.categories: Dict[VoiceCategory, List[str]] = {}
        self.languages: Dict[str, List[str]] = {}
        self.engines: Dict[VoiceEngine, List[str]] = {}
        self.logger = self._setup_logging()
        
        # Initialize voice library
        self._discover_all_voices()
        self._categorize_voices()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('VoiceLibrary')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _discover_all_voices(self):
        """Discover voices from all available engines"""
        self.logger.info("🔍 Discovering voices from all engines...")
        
        # Discover VibeVoice voices
        vibevoice_voices = self._discover_vibevoice_voices()
        self.logger.info(f"✅ VibeVoice: {len(vibevoice_voices)} voices")
        
        # Discover Coqui voices
        coqui_voices = self._discover_coqui_voices()
        self.logger.info(f"✅ Coqui AI: {len(coqui_voices)} voices")
        
        # Discover demo voices
        demo_voices = self._discover_demo_voices()
        self.logger.info(f"✅ Demo samples: {len(demo_voices)} voices")
        
        # Discover custom/cloned voices
        custom_voices = self._discover_custom_voices()
        self.logger.info(f"✅ Custom voices: {len(custom_voices)} voices")
        
        total_voices = len(vibevoice_voices) + len(coqui_voices) + len(demo_voices) + len(custom_voices)
        self.logger.info(f"📊 Total voices discovered: {total_voices}")
    
    def _discover_vibevoice_voices(self) -> List[VoiceInfo]:
        """Discover VibeVoice model voices"""
        voices: List[VoiceInfo] = []
        
        # VibeVoice models with different configurations
        vibevoice_models = [
            {
                "id": "vibevoice_1.5b_default",
                "name": "VibeVoice 1.5B Default",
                "model_path": "microsoft/VibeVoice-1.5B",
                "description": "High-quality conversational voice with natural prosody",
                "quality": "high"
            },
            {
                "id": "vibevoice_large_default",
                "name": "VibeVoice Large Default",
                "model_path": "aoi-ot/VibeVoice-Large",
                "description": "Enhanced quality large model with superior consistency",
                "quality": "premium"
            }
        ]
        
        demo_voice_dir = Path("demo/voices")
        voice_samples = []
        if demo_voice_dir.exists():
            voice_samples = list(demo_voice_dir.glob("*.wav")) + list(demo_voice_dir.glob("*.mp3"))
        
        if voice_samples:
            for model in vibevoice_models:
                for sample in voice_samples:
                    voice_info = self._parse_voice_filename(sample.name)
                    voice = VoiceInfo(
                        id=f"{model['id']}_{voice_info['code']}",
                        name=f"{model['name']} ({voice_info['name']})",
                        display_name=voice_info['name'],
                        language=voice_info['language'],
                        language_code=voice_info['lang_code'],
                        country=voice_info['country'],
                        gender=voice_info['gender'],
                        age="adult",
                        style="conversational",
                        quality=model['quality'],
                        engine="vibevoice",
                        model_path=model['model_path'],
                        description=f"{model['description']} - {voice_info['name']} voice",
                        sample_url=str(sample),
                        tags=["vibevoice", "conversational", voice_info['language']],
                        accent=voice_info.get('accent', 'neutral')
                    )
                    voices.append(voice)
                    self.voices[voice.id] = voice
        else:
            # Fallback: seed a minimal catalog so that the GUI always has base VibeVoice entries
            default_meta = {
                "language": "English",
                "language_code": "en",
                "country": "US",
                "gender": "neutral",
                "accent": "neutral",
                "style": "conversational",
            }
            for model in vibevoice_models:
                voice = VoiceInfo(
                    id=model['id'],
                    name=model['name'],
                    display_name=model['name'],
                    language=default_meta["language"],
                    language_code=default_meta["language_code"],
                    country=default_meta["country"],
                    gender=default_meta["gender"],
                    age="adult",
                    style=default_meta["style"],
                    quality=model['quality'],
                    engine="vibevoice",
                    model_path=model['model_path'],
                    description=model['description'],
                    sample_url=None,
                    tags=["vibevoice", "conversational", default_meta["language"]],
                    accent=default_meta["accent"],
                )
                voices.append(voice)
                self.voices[voice.id] = voice
        
        return voices
    
    def _discover_coqui_voices(self) -> List[VoiceInfo]:
        """Discover Coqui AI model voices"""
        voices = []
        
        # Comprehensive list of Coqui TTS models with metadata
        coqui_models = [
            # English voices
            {"name": "LJSpeech Tacotron2", "model": "tts_models/en/ljspeech/tacotron2-DDC", "lang": "en", "gender": "female", "quality": "high", "style": "clear"},
            {"name": "LJSpeech GlowTTS", "model": "tts_models/en/ljspeech/glow-tts", "lang": "en", "gender": "female", "quality": "high", "style": "natural"},
            {"name": "LJSpeech VITS", "model": "tts_models/en/ljspeech/vits", "lang": "en", "gender": "female", "quality": "premium", "style": "expressive"},
            {"name": "LJSpeech FastPitch", "model": "tts_models/en/ljspeech/fast_pitch", "lang": "en", "gender": "female", "quality": "high", "style": "fast"},
            {"name": "VCTK VITS", "model": "tts_models/en/vctk/vits", "lang": "en", "gender": "mixed", "quality": "premium", "style": "multi-speaker"},
            {"name": "Jenny Voice", "model": "tts_models/en/jenny/jenny", "lang": "en", "gender": "female", "quality": "premium", "style": "professional"},
            {"name": "Tortoise V2", "model": "tts_models/en/multi-dataset/tortoise-v2", "lang": "en", "gender": "mixed", "quality": "premium", "style": "expressive"},
            
            # Multilingual models
            {"name": "XTTS v2", "model": "tts_models/multilingual/multi-dataset/xtts_v2", "lang": "multilingual", "gender": "cloneable", "quality": "premium", "style": "voice-cloning"},
            {"name": "YourTTS", "model": "tts_models/multilingual/multi-dataset/your_tts", "lang": "multilingual", "gender": "mixed", "quality": "high", "style": "multi-speaker"},
            {"name": "Bark", "model": "tts_models/multilingual/multi-dataset/bark", "lang": "multilingual", "gender": "mixed", "quality": "premium", "style": "expressive"},
            
            # Spanish voices
            {"name": "Spanish Mai", "model": "tts_models/es/mai/tacotron2-DDC", "lang": "es", "gender": "female", "quality": "high", "style": "clear"},
            {"name": "Spanish CSS10", "model": "tts_models/es/css10/vits", "lang": "es", "gender": "female", "quality": "high", "style": "natural"},
            
            # French voices
            {"name": "French Mai", "model": "tts_models/fr/mai/tacotron2-DDC", "lang": "fr", "gender": "female", "quality": "high", "style": "clear"},
            {"name": "French CSS10", "model": "tts_models/fr/css10/vits", "lang": "fr", "gender": "female", "quality": "high", "style": "natural"},
            
            # German voices
            {"name": "Thorsten Tacotron2", "model": "tts_models/de/thorsten/tacotron2-DDC", "lang": "de", "gender": "male", "quality": "high", "style": "clear"},
            {"name": "Thorsten VITS", "model": "tts_models/de/thorsten/vits", "lang": "de", "gender": "male", "quality": "premium", "style": "natural"},
            {"name": "German CSS10", "model": "tts_models/de/css10/vits-neon", "lang": "de", "gender": "female", "quality": "high", "style": "bright"},
            
            # Italian voices
            {"name": "Italian Mai Female", "model": "tts_models/it/mai_female/vits", "lang": "it", "gender": "female", "quality": "high", "style": "natural"},
            {"name": "Italian Mai Male", "model": "tts_models/it/mai_male/vits", "lang": "it", "gender": "male", "quality": "high", "style": "natural"},
            
            # Japanese voices
            {"name": "Kokoro", "model": "tts_models/ja/kokoro/tacotron2-DDC", "lang": "ja", "gender": "female", "quality": "high", "style": "clear"},
            
            # Chinese voices
            {"name": "Baker Chinese", "model": "tts_models/zh-CN/baker/tacotron2-DDC-GST", "lang": "zh", "gender": "female", "quality": "high", "style": "expressive"},
            
            # Portuguese voices
            {"name": "Portuguese CV", "model": "tts_models/pt/cv/vits", "lang": "pt", "gender": "mixed", "quality": "high", "style": "natural"},
            
            # Dutch voices
            {"name": "Dutch Mai", "model": "tts_models/nl/mai/tacotron2-DDC", "lang": "nl", "gender": "female", "quality": "high", "style": "clear"},
            {"name": "Dutch CSS10", "model": "tts_models/nl/css10/vits", "lang": "nl", "gender": "female", "quality": "high", "style": "natural"},
            
            # Nordic languages
            {"name": "Finnish CSS10", "model": "tts_models/fi/css10/vits", "lang": "fi", "gender": "female", "quality": "high", "style": "natural"},
            {"name": "Swedish CV", "model": "tts_models/sv/cv/vits", "lang": "sv", "gender": "mixed", "quality": "high", "style": "natural"},
            {"name": "Danish CV", "model": "tts_models/da/cv/vits", "lang": "da", "gender": "mixed", "quality": "high", "style": "natural"},
            
            # Slavic languages
            {"name": "Ukrainian Mai", "model": "tts_models/uk/mai/vits", "lang": "uk", "gender": "female", "quality": "high", "style": "natural"},
            {"name": "Polish Female", "model": "tts_models/pl/mai_female/vits", "lang": "pl", "gender": "female", "quality": "high", "style": "natural"},
            {"name": "Czech CV", "model": "tts_models/cs/cv/vits", "lang": "cs", "gender": "mixed", "quality": "high", "style": "natural"},
            {"name": "Bulgarian CV", "model": "tts_models/bg/cv/vits", "lang": "bg", "gender": "mixed", "quality": "high", "style": "natural"},
            
            # Other European languages
            {"name": "Hungarian CSS10", "model": "tts_models/hu/css10/vits", "lang": "hu", "gender": "female", "quality": "high", "style": "natural"},
            {"name": "Greek CV", "model": "tts_models/el/cv/vits", "lang": "el", "gender": "mixed", "quality": "high", "style": "natural"},
            {"name": "Turkish Glow", "model": "tts_models/tr/common-voice/glow-tts", "lang": "tr", "gender": "mixed", "quality": "high", "style": "clear"},
            
            # African languages
            {"name": "Yoruba Bible", "model": "tts_models/yor/openbible/vits", "lang": "yo", "gender": "mixed", "quality": "good", "style": "narrative"},
            {"name": "Hausa Bible", "model": "tts_models/hau/openbible/vits", "lang": "ha", "gender": "mixed", "quality": "good", "style": "narrative"},
            
            # South Asian languages
            {"name": "Bengali Male", "model": "tts_models/bn/custom/vits-male", "lang": "bn", "gender": "male", "quality": "high", "style": "clear"},
            {"name": "Bengali Female", "model": "tts_models/bn/custom/vits-female", "lang": "bn", "gender": "female", "quality": "high", "style": "clear"},
        ]
        
        for model_info in coqui_models:
            voice_id = f"coqui_{model_info['name'].lower().replace(' ', '_').replace('-', '_')}"
            
            voice = VoiceInfo(
                id=voice_id,
                name=model_info['name'],
                display_name=model_info['name'],
                language=self._get_language_name(model_info['lang']),
                language_code=model_info['lang'],
                country=self._get_country_from_lang(model_info['lang']),
                gender=model_info['gender'],
                age="adult",
                style=model_info['style'],
                quality=model_info['quality'],
                engine="coqui",
                model_path=model_info['model'],
                description=f"Coqui AI {model_info['name']} - {model_info['style']} style",
                tags=["coqui", model_info['style'], model_info['lang']],
                is_premium=(model_info['quality'] == "premium")
            )
            
            voices.append(voice)
            self.voices[voice.id] = voice
        
        return voices
    
    def _discover_demo_voices(self) -> List[VoiceInfo]:
        """Discover voices from demo samples"""
        voices = []
        demo_dir = Path("demo/voices")
        
        if not demo_dir.exists():
            return voices
        
        voice_files = list(demo_dir.glob("*.wav")) + list(demo_dir.glob("*.mp3"))
        
        for voice_file in voice_files:
            voice_info = self._parse_voice_filename(voice_file.name)
            
            voice = VoiceInfo(
                id=f"demo_{voice_info['code']}",
                name=f"Demo {voice_info['name']}",
                display_name=voice_info['name'],
                language=voice_info['language'],
                language_code=voice_info['lang_code'],
                country=voice_info['country'],
                gender=voice_info['gender'],
                age="adult",
                style="demo",
                quality="sample",
                engine="demo",
                model_path=str(voice_file),
                description=f"Demo voice sample - {voice_info['name']}",
                sample_url=str(voice_file),
                tags=["demo", "sample", voice_info['language']]
            )
            
            voices.append(voice)
            self.voices[voice.id] = voice
        
        return voices
    
    def _discover_custom_voices(self) -> List[VoiceInfo]:
        """Discover custom/cloned voices"""
        voices = []
        custom_dir = Path("voices/custom")
        
        if not custom_dir.exists():
            return voices
        
        # Look for voice configuration files
        config_files = list(custom_dir.glob("*.json"))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    voice_config = json.load(f)
                
                voice = VoiceInfo(
                    id=f"custom_{voice_config['id']}",
                    name=voice_config['name'],
                    display_name=voice_config.get('display_name', voice_config['name']),
                    language=voice_config.get('language', 'English'),
                    language_code=voice_config.get('language_code', 'en'),
                    country=voice_config.get('country', 'US'),
                    gender=voice_config.get('gender', 'neutral'),
                    age=voice_config.get('age', 'adult'),
                    style=voice_config.get('style', 'custom'),
                    quality=voice_config.get('quality', 'custom'),
                    engine="custom",
                    model_path=voice_config.get('model_path', ''),
                    description=voice_config.get('description', 'Custom voice'),
                    sample_url=voice_config.get('sample_url'),
                    is_clone=voice_config.get('is_clone', True),
                    tags=voice_config.get('tags', ['custom'])
                )
                
                voices.append(voice)
                self.voices[voice.id] = voice
                
            except Exception as e:
                self.logger.warning(f"Failed to load custom voice config {config_file}: {e}")
        
        return voices
    
    def _parse_voice_filename(self, filename: str) -> Dict[str, str]:
        """Parse voice characteristics from filename"""
        # Default values
        info = {
            'code': filename.replace('.wav', '').replace('.mp3', ''),
            'name': 'Unknown',
            'language': 'English',
            'lang_code': 'en',
            'country': 'US',
            'gender': 'neutral'
        }
        
        # Parse format like: en-Alice_woman.wav, zh-Anchen_man_bgm.wav
        if '-' in filename:
            parts = filename.split('-')
            if len(parts) >= 2:
                lang_part = parts[0]
                name_part = parts[1].replace('.wav', '').replace('.mp3', '')
                
                # Language mapping
                lang_map = {
                    'en': ('English', 'en', 'US'),
                    'zh': ('Chinese', 'zh', 'CN'),
                    'in': ('Hindi', 'hi', 'IN'),
                    'fr': ('French', 'fr', 'FR'),
                    'de': ('German', 'de', 'DE'),
                    'es': ('Spanish', 'es', 'ES'),
                    'ja': ('Japanese', 'ja', 'JP'),
                    'pt': ('Portuguese', 'pt', 'PT'),
                    'it': ('Italian', 'it', 'IT')
                }
                
                if lang_part in lang_map:
                    info['language'], info['lang_code'], info['country'] = lang_map[lang_part]
                
                # Extract name and gender
                if '_' in name_part:
                    name_parts = name_part.split('_')
                    info['name'] = name_parts[0].title()
                    
                    # Detect gender from keywords
                    gender_parts = [p.lower() for p in name_parts[1:]]
                    if any(g in gender_parts for g in ['woman', 'female', 'girl']):
                        info['gender'] = 'female'
                    elif any(g in gender_parts for g in ['man', 'male', 'boy']):
                        info['gender'] = 'male'
                else:
                    info['name'] = name_part.title()
        
        return info
    
    def _get_language_name(self, code: str) -> str:
        """Convert language code to full name"""
        lang_map = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ru': 'Russian',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
            'pl': 'Polish',
            'cs': 'Czech',
            'hu': 'Hungarian',
            'tr': 'Turkish',
            'el': 'Greek',
            'uk': 'Ukrainian',
            'bg': 'Bulgarian',
            'hr': 'Croatian',
            'sk': 'Slovak',
            'sl': 'Slovenian',
            'et': 'Estonian',
            'lv': 'Latvian',
            'lt': 'Lithuanian',
            'mt': 'Maltese',
            'ga': 'Irish',
            'cy': 'Welsh',
            'is': 'Icelandic',
            'mk': 'Macedonian',
            'sq': 'Albanian',
            'sr': 'Serbian',
            'bs': 'Bosnian',
            'me': 'Montenegrin',
            'multilingual': 'Multilingual'
        }
        return lang_map.get(code, code.upper())
    
    def _get_country_from_lang(self, code: str) -> str:
        """Get default country for language code"""
        country_map = {
            'en': 'US',
            'es': 'ES',
            'fr': 'FR', 
            'de': 'DE',
            'it': 'IT',
            'pt': 'PT',
            'zh': 'CN',
            'ja': 'JP',
            'ko': 'KR',
            'ru': 'RU',
            'ar': 'SA',
            'hi': 'IN',
            'nl': 'NL',
            'sv': 'SE',
            'da': 'DK',
            'no': 'NO',
            'fi': 'FI',
            'multilingual': 'GLOBAL'
        }
        return country_map.get(code, 'UNKNOWN')
    
    def _categorize_voices(self):
        """Categorize voices by type, language, engine, etc."""
        # Initialize categories
        for category in VoiceCategory:
            self.categories[category] = []
        
        for engine in VoiceEngine:
            self.engines[engine] = []
        
        # Categorize each voice
        for voice_id, voice in self.voices.items():
            # By style/category
            if voice.style in ['professional', 'clear', 'news']:
                self.categories[VoiceCategory.PROFESSIONAL].append(voice_id)
            elif voice.style in ['conversational', 'casual', 'friendly']:
                self.categories[VoiceCategory.CASUAL].append(voice_id)
            elif voice.style in ['narrative', 'storytelling', 'book']:
                self.categories[VoiceCategory.NARRATIVE].append(voice_id)
            elif voice.style in ['character', 'expressive', 'dramatic']:
                self.categories[VoiceCategory.CHARACTER].append(voice_id)
            elif voice.style in ['educational', 'instructional', 'tutorial']:
                self.categories[VoiceCategory.EDUCATIONAL].append(voice_id)
            else:
                self.categories[VoiceCategory.CONVERSATIONAL].append(voice_id)
            
            # By language
            if voice.language not in self.languages:
                self.languages[voice.language] = []
            self.languages[voice.language].append(voice_id)
            
            # By engine
            engine_enum = None
            if voice.engine == "vibevoice":
                engine_enum = VoiceEngine.VIBEVOICE
            elif voice.engine == "coqui":
                engine_enum = VoiceEngine.COQUI
            elif voice.engine in ["simple", "demo"]:
                engine_enum = VoiceEngine.SIMPLE
            elif voice.engine == "custom":
                engine_enum = VoiceEngine.CUSTOM
            
            if engine_enum:
                self.engines[engine_enum].append(voice_id)
    
    def get_all_voices(self) -> Dict[str, VoiceInfo]:
        """Get all available voices"""
        return self.voices.copy()
    
    def search_voices(self, query: str = "", language: str = "", 
                     gender: str = "", style: str = "", 
                     engine: str = "", quality: str = "") -> List[VoiceInfo]:
        """Search voices with filters"""
        results = []
        
        for voice in self.voices.values():
            # Text search in name and description
            if query and query.lower() not in voice.name.lower() and query.lower() not in voice.description.lower():
                continue
            
            # Filter by language
            if language and language.lower() not in voice.language.lower() and language.lower() != voice.language_code.lower():
                continue
            
            # Filter by gender
            if gender and gender.lower() != voice.gender.lower():
                continue
            
            # Filter by style
            if style and style.lower() not in voice.style.lower():
                continue
            
            # Filter by engine
            if engine and engine.lower() != voice.engine.lower():
                continue
            
            # Filter by quality
            if quality and quality.lower() != voice.quality.lower():
                continue
            
            results.append(voice)
        
        return sorted(results, key=lambda v: (v.quality == "premium", v.name))
    
    def get_voices_by_category(self, category: VoiceCategory) -> List[VoiceInfo]:
        """Get voices by category"""
        voice_ids = self.categories.get(category, [])
        return [self.voices[vid] for vid in voice_ids if vid in self.voices]
    
    def get_voices_by_language(self, language: str) -> List[VoiceInfo]:
        """Get voices by language"""
        voice_ids = self.languages.get(language, [])
        return [self.voices[vid] for vid in voice_ids if vid in self.voices]
    
    def get_voices_by_engine(self, engine: VoiceEngine) -> List[VoiceInfo]:
        """Get voices by engine"""
        voice_ids = self.engines.get(engine, [])
        return [self.voices[vid] for vid in voice_ids if vid in self.voices]
    
    def get_voice_by_id(self, voice_id: str) -> Optional[VoiceInfo]:
        """Get specific voice by ID"""
        return self.voices.get(voice_id)
    
    def get_recommended_voices(self, content_type: str = "general", 
                             language: str = "en") -> List[VoiceInfo]:
        """Get recommended voices for specific content type"""
        recommendations = []
        
        # Define content type preferences
        style_preferences = {
            "podcast": ["conversational", "professional", "engaging"],
            "audiobook": ["narrative", "storytelling", "clear"],
            "news": ["professional", "clear", "authoritative"],
            "educational": ["clear", "patient", "instructional"],
            "entertainment": ["expressive", "character", "dramatic"],
            "general": ["conversational", "natural", "clear"]
        }
        
        preferred_styles = style_preferences.get(content_type, style_preferences["general"])
        
        # Find voices matching preferences
        for voice in self.voices.values():
            if language.lower() in voice.language.lower() or language.lower() == voice.language_code.lower():
                if any(style in voice.style.lower() or style in voice.description.lower() for style in preferred_styles):
                    recommendations.append(voice)
        
        # Sort by quality and relevance
        return sorted(recommendations, key=lambda v: (v.quality == "premium", v.quality == "high", v.name))[:10]
    
    def export_voice_library(self, output_file: str):
        """Export voice library to JSON"""
        export_data = {
            "voices": {vid: asdict(voice) for vid, voice in self.voices.items()},
            "categories": {cat.value: voices for cat, voices in self.categories.items()},
            "languages": self.languages,
            "engines": {eng.value: voices for eng, voices in self.engines.items()},
            "statistics": {
                "total_voices": len(self.voices),
                "total_languages": len(self.languages),
                "total_engines": len([eng for eng, voices in self.engines.items() if voices])
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📁 Voice library exported to {output_file}")
    
    def get_voice_statistics(self) -> Dict:
        """Get comprehensive voice library statistics"""
        stats = {
            "total_voices": len(self.voices),
            "by_engine": {},
            "by_language": {},
            "by_gender": {"male": 0, "female": 0, "neutral": 0, "mixed": 0, "cloneable": 0},
            "by_quality": {"premium": 0, "high": 0, "good": 0, "sample": 0, "custom": 0},
            "by_category": {}
        }
        
        # Count by engine
        for engine, voice_ids in self.engines.items():
            stats["by_engine"][engine.value] = len(voice_ids)
        
        # Count by language
        for language, voice_ids in self.languages.items():
            stats["by_language"][language] = len(voice_ids)
        
        # Count by gender and quality
        for voice in self.voices.values():
            if voice.gender in stats["by_gender"]:
                stats["by_gender"][voice.gender] += 1
            
            if voice.quality in stats["by_quality"]:
                stats["by_quality"][voice.quality] += 1
        
        # Count by category
        for category, voice_ids in self.categories.items():
            stats["by_category"][category.value] = len(voice_ids)
        
        return stats

def main():
    """CLI interface for voice library management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage VibeVoice voice library")
    parser.add_argument("command", choices=["list", "search", "stats", "export"], help="Command to execute")
    parser.add_argument("-q", "--query", help="Search query")
    parser.add_argument("-l", "--language", help="Filter by language")
    parser.add_argument("-g", "--gender", help="Filter by gender")
    parser.add_argument("-s", "--style", help="Filter by style")
    parser.add_argument("-e", "--engine", help="Filter by engine")
    parser.add_argument("-o", "--output", help="Output file for export")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--limit", type=int, default=20, help="Limit results")
    
    args = parser.parse_args()
    
    # Initialize voice library
    library = EnhancedVoiceLibrary()
    
    if args.command == "list":
        voices = library.search_voices(
            query=args.query or "",
            language=args.language or "",
            gender=args.gender or "",
            style=args.style or "",
            engine=args.engine or ""
        )
        
        print(f"📋 Found {len(voices)} voices:")
        for i, voice in enumerate(voices[:args.limit]):
            print(f"{i+1:2d}. {voice.name}")
            print(f"    Language: {voice.language} | Gender: {voice.gender} | Engine: {voice.engine}")
            print(f"    Style: {voice.style} | Quality: {voice.quality}")
            if voice.description:
                print(f"    Description: {voice.description}")
            print()
    
    elif args.command == "search":
        if not args.query:
            print("❌ Search query required")
            return
        
        voices = library.search_voices(query=args.query)
        print(f"🔍 Search results for '{args.query}': {len(voices)} voices found")
        
        for voice in voices[:args.limit]:
            print(f"• {voice.name} ({voice.language}, {voice.engine})")
    
    elif args.command == "stats":
        stats = library.get_voice_statistics()
        
        print("📊 Voice Library Statistics")
        print(f"Total voices: {stats['total_voices']}")
        print()
        
        print("By Engine:")
        for engine, count in stats['by_engine'].items():
            print(f"  {engine}: {count}")
        print()
        
        print("By Language:")
        for lang, count in sorted(stats['by_language'].items()):
            print(f"  {lang}: {count}")
        print()
        
        print("By Gender:")
        for gender, count in stats['by_gender'].items():
            print(f"  {gender}: {count}")
        print()
        
        print("By Quality:")
        for quality, count in stats['by_quality'].items():
            print(f"  {quality}: {count}")
    
    elif args.command == "export":
        output_file = args.output or "voice_library.json"
        library.export_voice_library(output_file)
        print(f"✅ Voice library exported to {output_file}")

if __name__ == "__main__":
    main()

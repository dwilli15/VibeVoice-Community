"""Local fallback adapter that exposes voice library helpers expected by ebook_gui.

This module wraps the EnhancedVoiceLibrary utilities so that imports of
``vibevoice.voice_library`` continue to work even when the upstream package
has not bundled the voice metadata helpers yet.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional

from enhanced_voice_library import EnhancedVoiceLibrary, VoiceInfo


class CommunityVoiceLibrary:
    """Thin wrapper around EnhancedVoiceLibrary with helper lookups."""

    def __init__(self) -> None:
        self._library = EnhancedVoiceLibrary()

    @property
    def _voices(self) -> Dict[str, VoiceInfo]:
        return self._library.voices

    def refresh(self) -> None:
        """Rebuild the underlying catalog."""
        self._library = EnhancedVoiceLibrary()

    def get_all_voices(self) -> List[VoiceInfo]:
        return list(self._voices.values())

    def search_voices(self, query: str = "", **filters) -> List[VoiceInfo]:
        return self._library.search_voices(query=query, **filters)

    def get_voices_by_language(self, language: str) -> List[VoiceInfo]:
        return self._library.get_voices_by_language(language)

    def get_voices_by_gender(self, gender: str) -> List[VoiceInfo]:
        return self._library.search_voices(gender=gender)

    def get_voice_by_id(self, voice_id: str) -> Optional[VoiceInfo]:
        return self._library.get_voice_by_id(voice_id)

    def find_by_name(self, name: str) -> Optional[VoiceInfo]:
        for voice in self._voices.values():
            if voice.name == name or voice.name.lower() == name.lower():
                return voice
        return None


voice_library = CommunityVoiceLibrary()


def _strip_display_markup(value: str) -> str:
    """Remove emoji markers and repeated whitespace from GUI labels."""
    marker_chars = [
        "🎙", "🎵", "🗣", "⭐", "✨", "🔹", "👩", "👨", "🤖", "🌍", "📤", "️"
    ]
    cleaned = value
    for marker in marker_chars:
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


@lru_cache(maxsize=1)
def get_voice_selector_data() -> Dict[str, List[str]]:
    """Metadata buckets used by the GUI filters."""
    voices = voice_library.get_all_voices()
    languages = sorted({voice.language for voice in voices})
    genders = sorted({voice.gender for voice in voices})
    engines = sorted({voice.engine for voice in voices})
    qualities = sorted({voice.quality for voice in voices})

    return {
        "languages": ["All Languages", *languages],
        "genders": ["All Genders", *genders],
        "engines": ["All Engines", *engines],
        "qualities": ["All Qualities", *qualities],
    }


def resolve_voice_mapping(display_value: str) -> Optional[Dict[str, str]]:
    """Translate a GUI selection into a concrete voice description."""
    if not display_value:
        return None

    cleaned = _strip_display_markup(display_value)
    if not cleaned:
        return None

    candidate = voice_library.find_by_name(cleaned)
    if not candidate and " " in cleaned:
        # Some labels append additional hints; try the last token group
        candidate = voice_library.find_by_name(cleaned.split()[-1])

    if not candidate:
        return None

    return {
        "id": candidate.id,
        "name": candidate.name,
        "engine": candidate.engine,
        "model_path": candidate.model_path,
        "language": candidate.language,
        "gender": candidate.gender,
        "quality": candidate.quality,
    }


# Convenience delegates so the adapter mirrors the original API.
get_all_voices = voice_library.get_all_voices
search_voices = voice_library.search_voices
get_voices_by_language = voice_library.get_voices_by_language
get_voices_by_gender = voice_library.get_voices_by_gender
get_voice_by_id = voice_library.get_voice_by_id

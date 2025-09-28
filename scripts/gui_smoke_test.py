"""Quick environment smoke test for the VibeVoice Community GUI.

This script verifies that the critical Python modules import correctly,
voice metadata is available, and the Hugging Face cache is reachable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _check_vibevoice() -> List[str]:
    issues: List[str] = []
    try:
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig  # noqa: F401
        from vibevoice.modular.modeling_vibevoice_inference import (  # noqa: F401
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic path
        issues.append(f"VibeVoice import failed: {exc}")
    return issues


def _check_gui_flags() -> List[str]:
    issues: List[str] = []
    try:
        import ebook_gui  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic path
        issues.append(f"ebook_gui import failed: {exc}")
        return issues

    if not getattr(ebook_gui, "VIBEVOICE_AVAILABLE", False):
        issues.append("ebook_gui reports VIBEVOICE_AVAILABLE = False")
    if not getattr(ebook_gui, "VOICE_LIBRARY_AVAILABLE", False):
        issues.append("ebook_gui reports VOICE_LIBRARY_AVAILABLE = False")
    return issues


def _check_voice_library() -> List[str]:
    issues: List[str] = []
    try:
        from voice_library_adapter import get_voice_selector_data, voice_library
    except Exception as exc:  # pragma: no cover - diagnostic path
        issues.append(f"voice_library_adapter import failed: {exc}")
        return issues

    voices = voice_library.get_all_voices()
    if not voices:
        issues.append("Voice library returned no voices")
    else:
        selector = get_voice_selector_data()
        required_keys = {"languages", "genders", "engines", "qualities"}
        if not required_keys.issubset(selector):
            issues.append("Voice selector metadata missing expected keys")

    return issues


def _resolve_hf_home() -> Path:
    candidates = [
        os.environ.get("HF_HOME"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "D:/omen/models/hf/hub",
    ]
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    return Path()


def main() -> None:
    issues: List[str] = []
    issues.extend(_check_vibevoice())
    issues.extend(_check_gui_flags())
    issues.extend(_check_voice_library())

    hf_home = _resolve_hf_home()
    if not hf_home:
        issues.append("Hugging Face cache not found; set HF_HOME to D:/omen/models/hf/hub")

    if issues:
        print("[FAIL] VibeVoice GUI smoke test found issues:")
        for item in issues:
            print(f" - {item}")
        raise SystemExit(1)

    from voice_library_adapter import voice_library  # late import to avoid mypy noise

    voices = voice_library.get_all_voices()
    print("[PASS] VibeVoice GUI smoke test passed")
    print(f"HF cache: {hf_home}")
    print(f"Discovered voices: {len(voices)}")
    sample = voices[0]
    print(f"Sample voice: {sample.name} ({sample.engine}, {sample.language})")


if __name__ == "__main__":
    main()

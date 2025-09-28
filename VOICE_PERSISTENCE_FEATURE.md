# Voice Persistence Feature

## Overview

The VibeVoice-Community Simple TTS Backend now supports persistent custom voice storage, allowing users to upload voice samples through the GUI and have them automatically saved for future use across sessions.

## Features

### 1. Custom Voice Storage
- **Location**: `%USERPROFILE%\AppData\Local\VibeVoice-Community\voices\`
- **Format**: Voice files are copied and renamed with timestamps for uniqueness
- **Metadata**: JSON configuration file stores voice metadata and file paths

### 2. Voice Persistence Methods

#### Save Custom Voice
```python
backend = SimpleTTSBackend()
success = backend.save_custom_voice(
    voice_file_path="path/to/voice.wav",
    voice_name="MyCustomVoice", 
    language="en",
    gender="female"
)
```

#### Get Custom Voices
```python
custom_voices = backend.get_custom_voices()
for voice in custom_voices:
    print(f"Voice: {voice.name} ({voice.language}, {voice.gender})")
    print(f"File: {voice.file_path}")
    print(f"Created: {voice.created_at}")
```

#### Remove Custom Voice
```python
success = backend.remove_custom_voice("MyCustomVoice")
```

### 3. Voice Selection Priority

The voice selection system now prioritizes voices in this order:
1. **VibeVoice** built-in voices (primary engine)
2. **Custom** uploaded voices
3. **eSpeak** voices (fallback)
4. **pyttsx3** voices (last resort)

### 4. Custom Voice Generation

Custom voices are processed using VibeVoice's neural TTS with the uploaded voice sample:
- Uses the custom voice file as a reference sample
- Generates speech with VibeVoice quality using the custom voice characteristics
- Falls back to silent audio if VibeVoice is unavailable

## File Structure

```
%USERPROFILE%\AppData\Local\VibeVoice-Community\
└── voices/
    ├── voice_library.json          # Voice metadata database
    ├── CustomVoice1_20250910_123456.wav
    ├── CustomVoice2_20250910_123457.wav
    └── ...
```

## Voice Library Format

The `voice_library.json` file stores metadata for all custom voices:

```json
[
  {
    "name": "MyCustomVoice",
    "language": "en",
    "gender": "female", 
    "engine": "custom",
    "file_path": "C:\\Users\\...\\voices\\MyCustomVoice_20250910_123456.wav",
    "created_at": "2025-09-10T12:34:56.123456"
  }
]
```

## Integration with GUI

For GUI applications, the voice persistence system provides:

1. **Voice Upload Handler**: Call `save_custom_voice()` when users upload voice files
2. **Voice List Updates**: Custom voices automatically appear in voice selection dropdowns
3. **Voice Management**: Users can remove unwanted custom voices through the GUI

### Example GUI Integration

```python
# When user uploads a voice file in GUI
def on_voice_upload(file_path, voice_name, language, gender):
    backend = SimpleTTSBackend()
    success = backend.save_custom_voice(file_path, voice_name, language, gender)
    if success:
        # Refresh voice list in GUI
        update_voice_dropdown()
        show_success_message(f"Voice '{voice_name}' saved successfully!")
    else:
        show_error_message("Failed to save voice. Please try again.")

# Refresh voice dropdown with custom voices
def update_voice_dropdown():
    backend = SimpleTTSBackend()
    all_voices = backend.get_voices()
    voice_names = [voice.name for voice in all_voices]
    populate_dropdown(voice_names)
```

## Benefits

1. **User Experience**: Custom voices persist across sessions
2. **Voice Library**: Build a personal collection of favorite voices
3. **Quality**: Custom voices use VibeVoice's neural TTS for high-quality output
4. **Flexibility**: Easy to add, remove, and manage custom voices
5. **Integration**: Seamless integration with existing voice selection system

## Technical Notes

- Voice files are validated before saving to ensure they exist
- Unique filenames prevent conflicts with timestamp-based naming
- Safe filename sanitization removes special characters
- Graceful fallback if custom voice files are moved or deleted
- Automatic cleanup of orphaned voice metadata

## Backward Compatibility

The voice persistence feature is fully backward compatible:
- Existing voice selection code continues to work
- Built-in VibeVoice voices remain as defaults
- No breaking changes to the API
- Custom voices enhance rather than replace existing functionality

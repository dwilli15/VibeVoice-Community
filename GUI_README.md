# VibeVoice Integrated TTS/STT Studio

A comprehensive Streamlit GUI that merges VibeVoice with Whisper, Audiblez, and Coqui TTS for an awesome one-stop shop for TTS and STT capabilities.

## 🚀 Features

### Text-to-Speech (TTS)
- **VibeVoice**: Long-form conversational TTS with multi-speaker support
- **Coqui TTS**: Advanced voice cloning and synthesis
- **Voice Presets**: Pre-configured voice samples for consistent output
- **Custom Voice Samples**: Upload your own voice references

### Speech-to-Text (STT)
- **OpenAI Whisper**: State-of-the-art speech transcription
- **Multiple Model Sizes**: From tiny (fast) to large (accurate)
- **Multiple Audio Formats**: Support for WAV, MP3, FLAC, M4A, OGG

### Advanced Features
- **Ebook to Audiobook**: Convert text files to full audiobooks
- **Podcast Generation**: Multi-speaker podcast creation with script parsing
- **Voice Cloning**: Clone voices from audio samples
- **Transcript Generation**: Convert audio to text with high accuracy

## 📋 Installation

### Method 1: Automatic Installation (Recommended)

1. First, ensure VibeVoice is installed:
```bash
pip install -e .
```

2. Run the GUI dependency installer:
```bash
python install_gui_deps.py
```

3. Launch the GUI:
```bash
streamlit run streamlit_gui.py
```

### Method 2: Manual Installation

1. Install VibeVoice:
```bash
pip install -e .
```

2. Install GUI dependencies:
```bash
pip install streamlit openai-whisper TTS soundfile pydub
```

3. Install FFmpeg (for better audio support):
   - **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

4. Launch the GUI:
```bash
streamlit run streamlit_gui.py
```

## 🎯 Usage

### Getting Started

1. **Launch the Application**:
   ```bash
   streamlit run streamlit_gui.py
   ```

2. **Load Models** (in the sidebar):
   - **VibeVoice**: Load the TTS model (default: microsoft/VibeVoice-1.5B)
   - **Whisper**: Choose model size (base recommended for balance)
   - **Coqui TTS**: Select from available TTS models

### Text-to-Speech

1. Navigate to the **"🗣️ Text-to-Speech"** tab
2. Enter your text in the text area
3. Choose your TTS engine (VibeVoice or Coqui TTS)
4. Select voice presets or upload custom voice samples
5. Click **"Generate Speech"** to create audio
6. Play the generated audio or download it

### Speech-to-Text

1. Navigate to the **"👂 Speech-to-Text"** tab
2. Upload an audio file (WAV, MP3, FLAC, M4A, OGG)
3. Click **"Transcribe Audio"** to generate transcript
4. View and download the transcript

### Ebook to Audiobook

1. Navigate to the **"📚 Ebook to Audiobook"** tab
2. Upload a text file or paste content directly
3. Configure chapter splitting (optional)
4. Select narrator voice
5. Click **"Convert to Audiobook"** to generate full audiobook
6. Download the complete audiobook file

### Podcast Generation

1. Navigate to the **"🎙️ Podcast Generation"** tab
2. Write or upload a script with speaker labels:
   ```
   Speaker 1: Welcome to our podcast!
   Speaker 2: Thanks for having me.
   Speaker 1: Today we're discussing...
   ```
3. Configure voices for each detected speaker
4. Click **"Generate Podcast"** to create multi-speaker audio
5. Download the generated podcast

### Voice Cloning

1. Navigate to the **"🎭 Voice Cloning"** tab
2. Upload a clean voice sample (3-10 seconds recommended)
3. Enter text to synthesize with the cloned voice
4. Choose cloning engine (VibeVoice or Coqui TTS)
5. Click **"Clone Voice and Generate"** to create speech
6. Download the cloned voice audio

## 🔧 Configuration

### Model Paths
- **VibeVoice**: Default is `microsoft/VibeVoice-1.5B`
- **Alternative**: Use `aoi-ot/VibeVoice-Large` for better quality
- **Local Models**: Specify local model paths if available

### Voice Presets
- Place voice samples in `demo/voices/` directory
- Supported formats: WAV files
- Files are automatically detected and loaded as presets

### Performance Optimization
- **GPU**: CUDA automatically detected if available
- **CPU Fallback**: Automatically uses CPU if CUDA unavailable
- **Model Size**: Choose appropriate Whisper model size for your needs

## 📁 File Structure

```
VibeVoice-Community/
├── streamlit_gui.py          # Main GUI application
├── install_gui_deps.py       # Dependency installer
├── requirements_gui.txt      # GUI dependencies
├── GUI_README.md            # This documentation
├── demo/
│   ├── voices/              # Voice preset directory
│   └── text_examples/       # Example scripts
└── vibevoice/               # Core VibeVoice modules
```

## 🎨 Features Overview

| Feature | VibeVoice | Whisper | Coqui TTS | Audiblez |
|---------|-----------|---------|-----------|----------|
| **Text-to-Speech** | ✅ | ❌ | ✅ | 🔄 Planned |
| **Speech-to-Text** | ❌ | ✅ | ❌ | ❌ |
| **Voice Cloning** | ✅ | ❌ | ✅ | 🔄 Planned |
| **Multi-Speaker** | ✅ | ❌ | ❌ | 🔄 Planned |
| **Long-Form** | ✅ | ✅ | ❌ | 🔄 Planned |

## 🚨 Tips and Best Practices

### For Best TTS Quality:
- Use clean, high-quality voice samples for cloning
- Keep voice samples between 3-10 seconds
- Use the Large VibeVoice model for better stability
- Add appropriate punctuation for natural speech patterns

### For Best STT Results:
- Use clear, noise-free audio recordings
- Choose appropriate Whisper model size:
  - **tiny/base**: Fast, good for simple speech
  - **small/medium**: Balanced speed and accuracy
  - **large**: Best accuracy, slower processing

### System Requirements:
- **Minimum**: 8GB RAM, CPU-only operation
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM
- **Storage**: 5-10GB for models (downloaded automatically)

## 🐛 Troubleshooting

### Common Issues:

1. **"Model not loaded" error**:
   - Click the "Load [Model Name]" button in the sidebar
   - Check internet connection for model downloads
   - Verify sufficient disk space

2. **Audio playback issues**:
   - Ensure browser supports audio playback
   - Check if generated audio file is valid
   - Try downloading and playing externally

3. **Memory errors**:
   - Reduce batch size or text length
   - Use smaller models (e.g., Whisper base instead of large)
   - Close other applications to free memory

4. **Installation issues**:
   - Run `python install_gui_deps.py` for automatic setup
   - Install FFmpeg for audio format support
   - Check Python version compatibility (3.9+)

### Getting Help:
- Check the [VibeVoice GitHub Issues](https://github.com/microsoft/VibeVoice/issues)
- Review model documentation on Hugging Face
- Ensure all dependencies are properly installed

## 📄 License

This GUI extends the VibeVoice project and inherits its licensing terms. Please refer to the main repository's LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with your improvements

## 🌟 Credits

- **VibeVoice**: Microsoft Research
- **Whisper**: OpenAI
- **Coqui TTS**: Coqui AI Team
- **Streamlit**: Streamlit Inc.
- **Audiblez**: Kokoro TTS integration (planned)

---

**Enjoy creating amazing audio content with VibeVoice Integrated TTS/STT Studio!** 🎙️✨
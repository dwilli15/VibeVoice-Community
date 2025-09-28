<div align="center">

## 🎙️ VibeVoice: A Frontier Long Conversational Text-to-Speech Model
[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=microsoft)](https://microsoft.github.io/VibeVoice)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f)
[![Technical Report](https://img.shields.io/badge/Technical-Report-red?logo=adobeacrobatreader)](https://arxiv.org/pdf/2508.19205)
[![Colab](https://img.shields.io/badge/Run-Colab-orange?logo=googlecolab)](https://colab.research.google.com/github/akadoubleone/VibeVoice-Community/blob/main/demo/VibeVoice_colab.ipynb)
[![Live Playground](https://img.shields.io/badge/Live-Playground-green?logo=gradio)](https://aka.ms/VibeVoice-Demo)

</div>
<!-- <div align="center">
<img src="Figures/log.png" alt="VibeVoice Logo" width="200">
</div> -->

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="Figures/VibeVoice_logo_white.png">
  <img src="Figures/VibeVoice_logo.png" alt="VibeVoice Logo" width="300">
</picture>
</div>

VibeVoice is a novel framework designed for generating **expressive**, **long-form**, **multi-speaker** conversational audio, such as podcasts, from text. It addresses significant challenges in traditional Text-to-Speech (TTS) systems, particularly in scalability, speaker consistency, and natural turn-taking.

A core innovation of VibeVoice is its use of continuous speech tokenizers (Acoustic and Semantic) operating at an ultra-low frame rate of 7.5 Hz. These tokenizers efficiently preserve audio fidelity while significantly boosting computational efficiency for processing long sequences. VibeVoice employs a [next-token diffusion](https://arxiv.org/abs/2412.08635) framework, leveraging a Large Language Model (LLM) to understand textual context and dialogue flow, and a diffusion head to generate high-fidelity acoustic details.

The model can synthesize speech up to **90 minutes** long with up to **4 distinct speakers**, surpassing the typical 1-2 speaker limits of many prior models. 


<p align="left">
  <img src="Figures/MOS-preference.png" alt="MOS Preference Results" height="260px">
  <img src="Figures/VibeVoice.jpg" alt="VibeVoice Overview" height="250px" style="margin-right: 10px;">
</p>

### 🔥 News

- **[2025-08-26] 🎉 We Open Source the [VibeVoice-Large](https://huggingface.co/aoi-ot/VibeVoice-Large) model weights!**
- **[2025-08-28] 🎉 We provide a [Colab](https://colab.research.google.com/github/akadoubleone/VibeVoice-Community/blob/main/demo/VibeVoice_colab.ipynb) script for easy access to our model. Due to GPU memory limitations, only VibeVoice-1.5B is supported.**
- **[2025-09-10] 🎉 Added Desktop GUI and enhanced Docker support with automatic port cleanup and GPU optimization!**
- **[2025-11-07] 🎧 MAJOR UPDATE: Complete Ebook to Audiobook conversion system with Voice Library (50+ voices), Chapter Selection, and Multi-format support!**

### 🎧 **NEW: Ebook to Audiobook Converter**

Transform your digital books into high-quality audiobooks with our comprehensive conversion system featuring:

#### 🎙️ **Voice Library (50+ Voices)**
- **9 Languages**: English (US/UK), Spanish, French, Hindi, Italian, Japanese, Portuguese, Chinese
- **Voice Categories**: Professional, Expressive, Storytelling, Premium voices
- **Smart Search**: Find voices by name, accent, or characteristics
- **Filter & Browse**: By language, gender, quality, and engine

#### 📖 **Smart Chapter Selection**
- **Interactive Chapter Picker**: Select specific chapters instead of converting entire books
- **Bulk Operations**: Select all, ranges, first N chapters
- **Chapter Preview**: See word counts and estimated duration
- **Smart Detection**: Automatic chapter boundary detection

#### 🔧 **Advanced Features**
- **Multiple Formats**: WAV (uncompressed), MP3 (chapters + combined), M4B (audiobook with metadata)
- **Metadata Support**: Embedded titles, authors, cover art for M4B files
- **Multi-Engine Support**: VibeVoice (primary), Coqui TTS (Python 3.11), Auto-selection
- **Quality Controls**: Adjustable speed (0.5x-2.0x), bitrate selection, advanced voice parameters

#### 📁 **Supported Input Formats**
- **PDF**: Portable Document Format
- **EPUB**: Electronic Publication Format (best for chapter detection)
- **DOCX**: Microsoft Word Documents
- **TXT**: Plain Text Files

#### 🚀 **Quick Start Ebook Conversion**
```bash
# Launch Ebook Converter GUI
docker-compose up -d vibe-ebook
# Access at http://localhost:7862

# Or run directly
python ebook_gui.py --port 7862
```

### 🖥️ User Interfaces

VibeVoice now provides multiple ways to interact with the model:

#### 🚀 Quick Start (Windows)
```bash
# Run the interactive launcher
start.bat
```

#### 🎧 **Ebook to Audiobook Converter**
- **50+ Voice Library** across 9 languages with smart search
- **Interactive Chapter Selection** with preview and bulk operations  
- **Multiple Output Formats** (WAV/MP3/M4B) with metadata support
- **Advanced Controls** for speed, quality, and voice parameters

```bash
# Web interface (recommended)
python ebook_gui.py --port 7862

# Docker deployment
docker-compose up -d vibe-ebook
# Access at http://localhost:7862

### Notes: voice samples and precision

- Auto-created demo voice sample:
  - If `demo/voices` is empty, the backend auto-creates a tiny placeholder sample at `demo/voices/en-Alice_woman.wav` so VibeVoice can run without manual reference audio. You can replace this with your own `.wav` or `.mp3` files; matching filenames (e.g., `en-Alice_woman.wav`) will be picked automatically.

- CUDA precision (bf16 vs fp16):
  - Some GPU/PyTorch builds don’t support bfloat16. We default to float16 on CUDA to avoid "unsupported ScalarType BFloat16" errors. To force bf16 on capable hardware, set `VIBEVOICE_USE_BF16=1` in your environment before running.

- External tokenizer warning (optional):
  - If an external tokenizer folder is configured (e.g., `D:/omen/temp/tokenizer`), you may see a class mismatch warning. It’s harmless for basic tests. Remove or update the external tokenizer path if you want to silence it.
```

#### 🖥️ Desktop Application
- **Native GUI** with advanced features
- **Batch processing** capabilities  
- **Voice management** tools
- **Real-time progress** tracking

```bash
python desktop_gui.py
```

#### 🌐 Web Interface (Speech Synthesis)
- **Browser-based** interface with streaming
- **Public sharing** via Gradio
- **Real-time generation** with progress updates
- **Docker containerized** for easy deployment

```bash
python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B --share
```

#### 🐳 Docker Setup (Improved)
- **GPU-optimized** containers with RTX 5070 support
- **Automatic port cleanup** to prevent conflicts
- **Easy management** scripts for Windows and Linux
- **Dual Python environments** (3.13 + 3.11 for TTS engines)

```bash
# Windows PowerShell
.\manage-container.ps1 start          # Main VibeVoice (port 7860)
.\manage-container.ps1 ebook          # Ebook Converter (port 7862)  
.\manage-container.ps1 ebook-py311    # Ebook + Coqui TTS (port 7863)

# Linux/Mac
./manage-container.sh start
./manage-container.sh ebook
```

# Docker Compose
docker-compose up --build -d
```

#### 📚 Ebook to Audiobook Converter
Transform any ebook into a professional audiobook with chapter markers and metadata!

**Supported Formats:**
- **PDF** - Portable Document Format
- **TXT** - Plain text files  
- **DOCX** - Microsoft Word documents
- **EPUB** - Electronic publication format

**Output Formats:**
- **WAV** - Uncompressed audio (best quality)
- **MP3** - Compressed audio with metadata
- **M4B** - Audiobook format with chapter markers

**TTS Engines:**
- **VibeVoice** - High-quality Microsoft TTS (always available)
- **Coqui AI** - Open-source TTS with multiple voices (Python 3.11 container)

```bash
# Start basic ebook converter (VibeVoice only)
.\manage-container.ps1 ebook
# Access at http://localhost:7862

# Start full converter with Coqui AI support
.\manage-container.ps1 ebook-py311
# Access at http://localhost:7863

# Command line usage
python ebook_converter.py input.pdf -o output/ --format m4b --title "My Book" --author "Author Name"

# Test the functionality
python test_packaging.py
```

**Features:**
- 🎭 **Multi-engine TTS** (VibeVoice + Coqui AI)
- 📖 **Smart chapter detection** from document structure
- 🎵 **Multiple formats** (WAV, MP3, M4B) with metadata
- 🖼️ **Cover art embedding** for M4B audiobooks
- ⚡ **Preview mode** for testing (first 2 chapters)
- 🔧 **Batch processing** with progress tracking
- 📱 **Chapter markers** for navigation
- 🎛️ **Configurable quality** (bitrate, speed, voice)

### 📋 TODO

- [ ] Merge models into official Hugging Face repository ([PR](https://github.com/huggingface/transformers/pull/40546))
- [ ] Release example training code and documentation
- [ ] VibePod:  End-to-end solution that creates podcasts from documents, webpages, or even a simple topic.

### 🎵 Demo Examples


**Video Demo**

We produced this video with [Wan2.2](https://github.com/Wan-Video/Wan2.2). We sincerely appreciate the Wan-Video team for their great work.

## Troubleshooting

If you encounter a NumPy ABI error on Windows (e.g., "numpy.dtype size changed" when importing transformers/pandas/sklearn), install the dev-tested stack first:

```
pip install -r dev-requirements.txt
```

**English**
<div align="center">

https://github.com/user-attachments/assets/0967027c-141e-4909-bec8-091558b1b784

</div>


**Chinese**
<div align="center">

https://github.com/user-attachments/assets/322280b7-3093-4c67-86e3-10be4746c88f

</div>

**Cross-Lingual**
<div align="center">

https://github.com/user-attachments/assets/838d8ad9-a201-4dde-bb45-8cd3f59ce722

</div>

**Spontaneous Singing**
<div align="center">

https://github.com/user-attachments/assets/6f27a8a5-0c60-4f57-87f3-7dea2e11c730

</div>


**Long Conversation with 4 people**
<div align="center">

https://github.com/user-attachments/assets/a357c4b6-9768-495c-a576-1618f6275727

</div>

For more examples, see the [Project Page](https://microsoft.github.io/VibeVoice).

Try it on [Colab](https://colab.research.google.com/github/akadoubleone/VibeVoice-Community/blob/main/demo/VibeVoice_colab.ipynb) or [Demo](https://aka.ms/VibeVoice-Demo).

## 🎤 Voice Library

Our comprehensive voice library includes **50+ high-quality voices** across **9 languages**, carefully curated for audiobook narration:

### 🌍 **Language Coverage**
| Language | Voices | Highlights |
|----------|--------|------------|
| 🇺🇸 **English (US)** | 20 voices | Premium collection with diverse styles |
| 🇬🇧 **English (UK)** | 8 voices | Elegant British accents |
| 🇪🇸 **Spanish** | 3 voices | Clear Latin American pronunciation |
| 🇫🇷 **French** | 1 voice | Sophisticated Parisian accent |
| 🇮🇳 **Hindi** | 4 voices | Modern Indian English blend |
| 🇮🇹 **Italian** | 2 voices | Melodic Mediterranean charm |
| 🇯🇵 **Japanese** | 5 voices | Traditional and modern styles |
| 🇧🇷 **Portuguese** | 3 voices | Rich Brazilian pronunciation |
| 🇨🇳 **Chinese** | 8 voices | Mandarin with regional variations |

### ⭐ **Voice Categories**

**Premium Voices** - Enhanced emotional range and expressiveness
- Heart (US Female) - Emotional storytelling
- Sky (US Female) - Bright and uplifting
- Fenrir (US Male) - Dramatic narration
- Isabella (UK Female) - Aristocratic elegance

**Professional Voices** - Clear, consistent delivery
- Jessica (US Female) - Business presentations
- Michael (US Male) - Professional narration
- Daniel (UK Male) - Distinguished gentleman
- Nicola (IT Male) - Passionate Italian delivery

**Character Voices** - Unique personalities for specific content
- Puck (US Male) - Playful and energetic
- Santa (US Male) - Warm and jolly
- Gongitsune (JP Female) - Traditional Japanese storytelling
- Nezumi (JP Female) - Cute and youthful

### 🔍 **Smart Voice Discovery**

**Search Features:**
- **Text Search**: Find voices by name, accent, or characteristics
- **Language Filter**: Browse voices by specific languages
- **Gender Filter**: Male, female, or neutral voices
- **Quality Filter**: Premium, high, or standard quality
- **Style Tags**: Professional, storytelling, expressive, etc.

**Example Searches:**
```
"british elegant" → Isabella, Emma, Alice (UK)
"storytelling male" → Fenrir, Fable, Kumo
"professional female" → Jessica, Nicole, Sara
"warm friendly" → Bella, River, Santa
```

### 🎛️ **Voice Controls**

**Basic Controls:**
- **Speed**: 0.5x - 2.0x (recommended: 1.3x for audiobooks)
- **Quality**: Multiple bitrate options for size vs quality
- **Format**: WAV (uncompressed), MP3 (balanced), M4B (audiobook)

**Advanced Controls** (Coming Soon):
- **Expressiveness**: Control emotional range and variation
- **Consistency**: Balance between natural variation and stability
- **Pacing**: Fine-tune rhythm and pauses
- **Emphasis**: Adjust sentence and paragraph emphasis



## Models
| Model | Context Length | Generation Length |  Weight |
|-------|----------------|----------|----------|
| VibeVoice-0.5B-Streaming | - | - | On the way |
| VibeVoice-1.5B | 64K | ~90 min | [HF link](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| VibeVoice-Large| 32K | ~45 min | [HF link](https://huggingface.co/aoi-ot/VibeVoice-Large) |

## Installation
We recommend to use NVIDIA Deep Learning Container to manage the CUDA environment. 

1. Launch docker
```bash
# NVIDIA PyTorch Container 24.07 / 24.10 / 24.12 verified. 
# Later versions are also compatible.
sudo docker run --privileged --net=host --ipc=host --ulimit memlock=-1:-1 --ulimit stack=-1:-1 --gpus all --rm -it  nvcr.io/nvidia/pytorch:24.07-py3

## If flash attention is not included in your docker environment, you need to install it manually
## Refer to https://github.com/Dao-AILab/flash-attention for installation instructions
# pip install flash-attn --no-build-isolation
```

2. Install from github
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice/

pip install -e .
```

### Local Workspace Setup (Omen)

For the `D:\omen` layout used in this workspace, point the Community virtual environment at the upstream package and reuse the shared Hugging Face cache:

```powershell
# Clone or refresh the upstream source alongside the community wrapper
gh repo clone vibevoice-community/VibeVoice pipelines/voice/VibeVoice

Set-Location D:\omen\pipelines\voice\VibeVoice-Community
# Install upstream package into the community venv
.\.venv\Scripts\python -m pip install -e ..\VibeVoice
```

```powershell
# Ensure the GUI and smoke tests see the shared HF cache
$env:HF_HOME = 'D:\omen\models\hf\hub'
$env:HUGGINGFACE_HUB_CACHE = $env:HF_HOME
```

Run the lightweight smoke test any time dependencies change to verify imports before launching the GUI:

```powershell
.\.venv\Scripts\python scripts\gui_smoke_test.py
```

## Usages

### 🚨 Tips
We observed users may encounter occasional instability when synthesizing Chinese speech. We recommend:

- Using English punctuation even for Chinese text, preferably only commas and periods.
- Using the Large model variant, which is considerably more stable.
- If you found the generated voice speak too fast. Please try to chunk your text with multiple speaker turns with same speaker label.

We'd like to thank [PsiPi](https://huggingface.co/PsiPi) for sharing an interesting way for emotion control. Detials can be found via [discussion12](https://huggingface.co/microsoft/VibeVoice-1.5B/discussions/12).

### Usage 1: Launch Gradio demo
```bash
apt update && apt install ffmpeg -y # for demo

# For 1.5B model
python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B --share

# For Large model
python demo/gradio_demo.py --model_path aoi-ot/VibeVoice-Large --share
```

### Usage 2: Inference from files directly
```bash
# We provide some LLM generated example scripts under demo/text_examples/ for demo
# 1 speaker
python demo/inference_from_file.py --model_path aoi-ot/VibeVoice-Large --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice

# or more speakers
python demo/inference_from_file.py --model_path aoi-ot/VibeVoice-Large --txt_path demo/text_examples/2p_music.txt --speaker_names Alice Frank
```

## FAQ
#### Q1: Is this a pretrained model?
**A:** Yes, it's a pretrained model without any post-training or benchmark-specific optimizations. In a way, this makes VibeVoice very versatile and fun to use.

#### Q2: Randomly trigger Sounds / Music / BGM.
**A:** As you can see from our demo page, the background music or sounds are spontaneous. This means we can't directly control whether they are generated or not. The model is content-aware, and these sounds are triggered based on the input text and the chosen voice prompt.

Here are a few things we've noticed:
*   If the voice prompt you use contains background music, the generated speech is more likely to have it as well. (The Large model is quite stable and effective at this—give it a try on the demo!)
*   If the voice prompt is clean (no BGM), but the input text includes introductory words or phrases like "Welcome to," "Hello," or "However," background music might still appear.
*   Speaker voice related, using "Alice" results in random BGM than others (fixed).
*   In other scenarios, the Large model is more stable and has a lower probability of generating unexpected background music.

In fact, we intentionally decided not to denoise our training data because we think it's an interesting feature for BGM to show up at just the right moment. You can think of it as a little easter egg we left for you.

#### Q3: Text normalization?
**A:** We don't perform any text normalization during training or inference. Our philosophy is that a large language model should be able to handle complex user inputs on its own. However, due to the nature of the training data, you might still run into some corner cases.

#### Q4: Singing Capability.
**A:** Our training data **doesn't contain any music data**. The ability to sing is an emergent capability of the model (which is why it might sound off-key, even on a famous song like 'See You Again'). (The Large model is more likely to exhibit this than the 1.5B).

#### Q5: Some Chinese pronunciation errors.
**A:** The volume of Chinese data in our training set is significantly smaller than the English data. Additionally, certain special characters (e.g., Chinese quotation marks) may occasionally cause pronunciation issues.

#### Q6: Instability of cross-lingual transfer.
**A:** The model does exhibit strong cross-lingual transfer capabilities, including the preservation of accents, but its performance can be unstable. This is an emergent ability of the model that we have not specifically optimized. It's possible that a satisfactory result can be achieved through repeated sampling.

## Risks and limitations

While efforts have been made to optimize it through various techniques, it may still produce outputs that are unexpected, biased, or inaccurate. VibeVoice inherits any biases, errors, or omissions produced by its base model (specifically, Qwen2.5 1.5b in this release).
Potential for Deepfakes and Disinformation: High-quality synthetic speech can be misused to create convincing fake audio content for impersonation, fraud, or spreading disinformation. Users must ensure transcripts are reliable, check content accuracy, and avoid using generated content in misleading ways. Users are expected to use the generated content and to deploy the models in a lawful manner, in full compliance with all applicable laws and regulations in the relevant jurisdictions. It is best practice to disclose the use of AI when sharing AI-generated content.

English and Chinese only: Transcripts in languages other than English or Chinese may result in unexpected audio outputs.

Non-Speech Audio: The model focuses solely on speech synthesis and does not handle background noise, music, or other sound effects.

Overlapping Speech: The current model does not explicitly model or generate overlapping speech segments in conversations.

We do not recommend using VibeVoice in commercial or real-world applications without further testing and development. This model is intended for research and development purposes only. Please use responsibly.

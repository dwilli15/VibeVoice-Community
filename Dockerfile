# VibeVoice-Community GPU image
FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    TRANSFORMERS_FORCE_ATTENTION_IMPLEMENTATION=sdpa \
    WORKDIR=/workspace \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR ${WORKDIR}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential lsof \
    && rm -rf /var/lib/apt/lists/*

# Copy repo content
COPY . /workspace

# Install python deps (editable)
RUN pip install --upgrade pip && \
    if [ -f pyproject.toml ]; then pip install -e .; fi && \
    pip install gradio soundfile librosa

# Default ports for web demos
EXPOSE 7860

# mount outputs at /workspace/outputs
RUN mkdir -p /workspace/outputs

# Default command: open the gradio demo if present; else drop to bash
CMD ["bash","-lc","lsof -ti:7860 | xargs -r kill -9; python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B --share"]


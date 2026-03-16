# MuseTalk 1.5 RunPod Serverless Worker
# ======================================
# GPU requirement: 4GB+ VRAM (fp16) — runs on T4, L4, A10, RTX 4000 etc.
#
# Strategy: Install hf_xet for Xet storage support, then use huggingface-cli.
# Download each model component separately to isolate failures.

FROM runpod/base:0.6.2-cuda11.8.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV FFMPEG_PATH=/usr/bin/ffmpeg

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — PyTorch with CUDA 11.8
RUN pip install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install huggingface CLI with Xet support (required for large files on HF)
RUN pip install --no-cache-dir "huggingface_hub[cli,hf_xet]"

# Clone MuseTalk
RUN git clone https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk

WORKDIR /app/MuseTalk

# Install MuseTalk requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install mmlab packages (required for face detection/parsing)
RUN pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmcv>=2.0.1" && \
    mim install "mmdet>=3.1.0" && \
    mim install "mmpose>=1.1.0"

# Additional dependencies for our handler
RUN pip install --no-cache-dir \
    runpod \
    pyyaml \
    requests

# ── Download model weights ──
# Each download is a separate RUN layer so failures are isolated and cached layers reused.

# 1. MuseTalk weights (v1.0 + v1.5) — this is the big one (~3.4GB for v1.5 unet.pth)
RUN huggingface-cli download TMElyralab/MuseTalk \
    --local-dir models/ \
    --resume-download

# 2. SD VAE (from stabilityai — standard HF, no Xet issues)
RUN huggingface-cli download stabilityai/sd-vae-ft-mse \
    config.json diffusion_pytorch_model.bin \
    --local-dir models/sd-vae-ft-mse/

# 3. Whisper (from openai — standard HF)
RUN huggingface-cli download openai/whisper-tiny \
    --local-dir models/whisper/

# Verify critical files exist
RUN echo "=== Verifying model weights ===" && \
    ls -la models/musetalk/ && \
    ls -la models/musetalkV15/ && \
    ls -la models/dwpose/ && \
    ls -la models/face-parse-bisent/ && \
    ls -la models/sd-vae-ft-mse/ && \
    ls -la models/whisper/ && \
    echo "=== All weights verified ==="

# Copy handler
WORKDIR /app
COPY rp_handler.py /app/rp_handler.py

# Start handler
CMD ["python3", "-u", "/app/rp_handler.py"]
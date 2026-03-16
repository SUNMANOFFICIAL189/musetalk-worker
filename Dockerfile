# MuseTalk 1.5 RunPod Serverless Worker
# ======================================
# Builds a container with MuseTalk 1.5 + all dependencies + model weights
# for deployment as a RunPod Serverless endpoint.
#
# GPU requirement: 4GB+ VRAM (fp16) — runs on T4, L4, A10, RTX 4000 etc.
# This makes it extremely cheap on RunPod Serverless.

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
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

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

# Additional dependencies
RUN pip install --no-cache-dir \
    runpod \
    pyyaml \
    requests

# Download model weights
# MuseTalk core model (~400MB)
RUN mkdir -p models/musetalk && \
    wget -q -O models/musetalk/musetalk.json \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/models/musetalk/musetalk.json" && \
    wget -q -O models/musetalk/pytorch_model.bin \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/models/musetalk/pytorch_model.bin"

# DWPose model (~300MB)
RUN mkdir -p models/dwpose && \
    wget -q -O models/dwpose/dw-ll_ucoco_384.pth \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/models/dwpose/dw-ll_ucoco_384.pth"

# Face parsing models
RUN mkdir -p models/face-parse-bisent && \
    wget -q -O models/face-parse-bisent/79999_iter.pth \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/models/face-parse-bisent/79999_iter.pth" && \
    wget -q -O models/face-parse-bisent/resnet18-5c106cde.pth \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/models/face-parse-bisent/resnet18-5c106cde.pth"

# SD VAE
RUN mkdir -p models/sd-vae-ft-mse && \
    wget -q -O models/sd-vae-ft-mse/config.json \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json" && \
    wget -q -O models/sd-vae-ft-mse/diffusion_pytorch_model.bin \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin"

# Whisper tiny
RUN mkdir -p models/whisper && \
    wget -q -O models/whisper/tiny.pt \
    "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"

# Copy handler
WORKDIR /app
COPY rp_handler.py /app/rp_handler.py

# Start handler
CMD ["python3", "-u", "/app/rp_handler.py"]
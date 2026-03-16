# MuseTalk 1.5 RunPod Serverless Worker
# ======================================
# GPU requirement: 4GB+ VRAM (fp16)
#
# Strategy: Build image with code + deps only. Download model weights at
# first startup via entrypoint.sh. This avoids HuggingFace Xet/LFS issues
# in RunPod's Docker build environment. First cold start takes ~5 min
# (downloading weights), subsequent starts are instant if using a Network Volume.

FROM runpod/base:0.6.2-cuda11.8.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV FFMPEG_PATH=/usr/bin/ffmpeg
ENV MUSETALK_DIR=/app/MuseTalk
ENV MODELS_DIR=/app/MuseTalk/models

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    git-lfs \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Python dependencies — PyTorch with CUDA 11.8
RUN pip install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# HuggingFace hub with xet support
RUN pip install --no-cache-dir "huggingface_hub[cli]" hf_xet

# Clone MuseTalk repo (code only, no large model files)
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk

WORKDIR /app/MuseTalk

# Install MuseTalk requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install mmlab packages
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

# Copy our files
WORKDIR /app
COPY rp_handler.py /app/rp_handler.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Entrypoint downloads weights on first start, then launches handler
CMD ["/app/entrypoint.sh"]
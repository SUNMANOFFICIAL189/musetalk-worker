# MuseTalk 1.5 RunPod Serverless Worker
# ======================================
# GPU requirement: 4GB+ VRAM (fp16) — runs on T4, L4, A10, RTX 4000 etc.

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

# Install whisper subpackage (required for audio feature extraction)
RUN pip install --editable ./musetalk/whisper

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
    requests \
    "huggingface_hub[cli]"

# Download ALL model weights using huggingface-cli (handles Xet/LFS correctly)
# This downloads the entire TMElyralab/MuseTalk repo into models/
RUN huggingface-cli download TMElyralab/MuseTalk --local-dir models/

# Download SD VAE weights (separate repo)
RUN huggingface-cli download stabilityai/sd-vae-ft-mse \
    config.json diffusion_pytorch_model.bin \
    --local-dir models/sd-vae-ft-mse/

# Download Whisper weights for audio processing
RUN mkdir -p models/whisper && \
    huggingface-cli download openai/whisper-tiny \
    --local-dir models/whisper/

# Copy handler
WORKDIR /app
COPY rp_handler.py /app/rp_handler.py

# Start handler
CMD ["python3", "-u", "/app/rp_handler.py"]
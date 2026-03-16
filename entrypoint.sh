#!/bin/bash
set -e

MODELS_DIR="/app/MuseTalk/models"
MARKER_FILE="$MODELS_DIR/.weights_downloaded"

echo "=== MuseTalk 1.5 RunPod Worker Starting ==="

# Only download weights if not already present
if [ ! -f "$MARKER_FILE" ]; then
    echo "=== Downloading model weights (first run only) ==="
    
    cd /app/MuseTalk
    
    # Use Python to download via huggingface_hub (more reliable than CLI in containers)
    python3 -c "
from huggingface_hub import snapshot_download, hf_hub_download
import os

models_dir = '$MODELS_DIR'
os.makedirs(models_dir, exist_ok=True)

print('[1/3] Downloading MuseTalk weights...')
snapshot_download(
    repo_id='TMElyralab/MuseTalk',
    local_dir=models_dir,
    resume_download=True,
)
print('[1/3] MuseTalk weights done.')

print('[2/3] Downloading SD VAE weights...')
vae_dir = os.path.join(models_dir, 'sd-vae-ft-mse')
os.makedirs(vae_dir, exist_ok=True)
for fname in ['config.json', 'diffusion_pytorch_model.bin']:
    hf_hub_download(
        repo_id='stabilityai/sd-vae-ft-mse',
        filename=fname,
        local_dir=vae_dir,
    )
print('[2/3] SD VAE done.')

print('[3/3] Downloading Whisper weights...')
whisper_dir = os.path.join(models_dir, 'whisper')
os.makedirs(whisper_dir, exist_ok=True)
snapshot_download(
    repo_id='openai/whisper-tiny',
    local_dir=whisper_dir,
    resume_download=True,
)
print('[3/3] Whisper done.')

print('=== All weights downloaded successfully ===')
"

    # Mark as downloaded so we skip on next cold start
    touch "$MARKER_FILE"
    
    echo "=== Verifying weights ==="
    ls -la "$MODELS_DIR"/musetalk/ 2>/dev/null || echo "WARNING: musetalk/ missing"
    ls -la "$MODELS_DIR"/musetalkV15/ 2>/dev/null || echo "WARNING: musetalkV15/ missing"
    ls -la "$MODELS_DIR"/dwpose/ 2>/dev/null || echo "WARNING: dwpose/ missing"
    ls -la "$MODELS_DIR"/face-parse-bisent/ 2>/dev/null || echo "WARNING: face-parse-bisent/ missing"
    ls -la "$MODELS_DIR"/sd-vae-ft-mse/ 2>/dev/null || echo "WARNING: sd-vae-ft-mse/ missing"
    ls -la "$MODELS_DIR"/whisper/ 2>/dev/null || echo "WARNING: whisper/ missing"
    echo "=== Verification complete ==="
else
    echo "=== Weights already present, skipping download ==="
fi

echo "=== Starting RunPod handler ==="
exec python3 -u /app/rp_handler.py
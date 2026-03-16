"""
MuseTalk 1.5 RunPod Serverless Handler
=======================================
Takes an InfiniteTalk-generated video + original audio as input,
refines the lip sync using MuseTalk 1.5, and returns the result.

Deploy this as a RunPod Serverless endpoint alongside your InfiniteTalk endpoint.
"""

import runpod
import os
import sys
import subprocess
import tempfile
import base64
import requests
import uuid
import yaml
import time

# Global model references (loaded once at cold start)
MUSETALK_DIR = "/app/MuseTalk"
MODELS_DIR = f"{MUSETALK_DIR}/models"
FFMPEG_PATH = "/usr/bin/ffmpeg"


def download_file(url_or_b64, suffix, tmp_dir):
    """Download from URL or decode base64 to a temp file."""
    filepath = os.path.join(tmp_dir, f"{uuid.uuid4().hex}{suffix}")

    if url_or_b64.startswith("http"):
        resp = requests.get(url_or_b64, timeout=120)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(resp.content)
    elif url_or_b64.startswith("data:"):
        # Strip data URI prefix
        b64_data = url_or_b64.split(",", 1)[1]
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(b64_data))
    else:
        # Assume raw base64
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(url_or_b64))

    return filepath


def run_musetalk(video_path, audio_path, output_dir, bbox_shift=0, use_float16=True):
    """
    Run MuseTalk inference on the given video + audio.

    MuseTalk replaces the mouth region of the video with phoneme-accurate
    lip shapes driven by the audio, keeping the rest of the frame intact.
    """
    # Create inference config
    config = {
        "video_path": video_path,
        "audio_path": audio_path,
        "bbox_shift": bbox_shift,
    }
    config_path = os.path.join(output_dir, "inference_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Build command
    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--inference_config", config_path,
        "--result_dir", output_dir,
    ]
    if use_float16:
        cmd.append("--use_float16")

    # Run from MuseTalk directory
    env = os.environ.copy()
    env["FFMPEG_PATH"] = FFMPEG_PATH

    print(f"[MuseTalk] Running inference...")
    print(f"[MuseTalk] Command: {' '.join(cmd)}")
    start_time = time.time()

    result = subprocess.run(
        cmd,
        cwd=MUSETALK_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min timeout
    )

    elapsed = time.time() - start_time
    print(f"[MuseTalk] Inference completed in {elapsed:.1f}s")

    if result.returncode != 0:
        print(f"[MuseTalk] STDERR: {result.stderr}")
        raise RuntimeError(f"MuseTalk inference failed: {result.stderr[-500:]}")

    # Find the output video
    # MuseTalk saves to result_dir with a pattern like <video_name>_<audio_name>.mp4
    output_files = [
        f for f in os.listdir(output_dir)
        if f.endswith(".mp4") and f != os.path.basename(video_path)
    ]

    if not output_files:
        raise RuntimeError(f"No output video found in {output_dir}")

    # Return the most recently modified mp4
    output_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
        reverse=True,
    )
    return os.path.join(output_dir, output_files[0])


def handler(event):
    """
    RunPod handler function.

    Input:
        video_url (str): URL to the InfiniteTalk output video
        video_base64 (str): OR base64 encoded video
        audio_url (str): URL to the original audio
        audio_base64 (str): OR base64 encoded audio
        bbox_shift (int): Mouth region shift (-9 to 9). Default 0.
                          Positive = more open mouth, negative = less open.
        use_float16 (bool): Use fp16 for inference. Default True.

    Output:
        video (str): Base64 encoded refined video
    """
    job_input = event["input"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = os.path.join(tmp_dir, "output")
        os.makedirs(output_dir)

        # Download inputs
        video_source = job_input.get("video_url") or job_input.get("video_base64")
        audio_source = job_input.get("audio_url") or job_input.get("audio_base64")

        if not video_source:
            return {"error": "No video input provided (video_url or video_base64)"}
        if not audio_source:
            return {"error": "No audio input provided (audio_url or audio_base64)"}

        print(f"[Handler] Downloading video...")
        video_path = download_file(video_source, ".mp4", tmp_dir)

        print(f"[Handler] Downloading audio...")
        audio_path = download_file(audio_source, ".wav", tmp_dir)

        # Preprocess audio to WAV 16kHz mono (MuseTalk expects this)
        preprocessed_audio = os.path.join(tmp_dir, "audio_16k.wav")
        subprocess.run([
            FFMPEG_PATH, "-y", "-i", audio_path,
            "-ar", "16000", "-ac", "1",
            "-af", "silenceremove=1:0:-40dB:1:1:-40dB",
            preprocessed_audio,
        ], capture_output=True, check=True)
        audio_path = preprocessed_audio

        # Run MuseTalk
        bbox_shift = job_input.get("bbox_shift", 0)
        use_float16 = job_input.get("use_float16", True)

        try:
            output_video = run_musetalk(
                video_path=video_path,
                audio_path=audio_path,
                output_dir=output_dir,
                bbox_shift=bbox_shift,
                use_float16=use_float16,
            )
        except Exception as e:
            return {"error": f"MuseTalk inference failed: {str(e)}"}

        # Encode output
        print(f"[Handler] Encoding output video...")
        with open(output_video, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        video_size_mb = os.path.getsize(output_video) / (1024 * 1024)
        print(f"[Handler] Output video size: {video_size_mb:.1f}MB")

        return {
            "video": f"data:video/mp4;base64,{video_b64}",
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
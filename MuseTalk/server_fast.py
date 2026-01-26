import os
import subprocess
import torch
import sys
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# --- CONFIGURATION ---
# 1. Base Paths (Dynamic - Finds folders automatically)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # This is .../MuseTalk
MUSETALK_DIR = BASE_DIR
# Assumes VoiceEngine is a sibling folder (../VoiceEngine)
VOICE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../VoiceEngine"))

# 2. Voice Engine Python
# Check if we should use the venv or system python
if os.path.exists(os.path.join(VOICE_DIR, "venv/bin/python")):
    VOICE_ENV_PYTHON = os.path.join(VOICE_DIR, "venv/bin/python")
else:
    # Fallback to system python if venv is missing
    VOICE_ENV_PYTHON = sys.executable

VOICE_SCRIPT = "commercial_voice.py" 
RAW_AUDIO_SOURCE = os.path.join(VOICE_DIR, "output_raw.wav")

# 3. MuseTalk Paths
# Ensure the audio directory exists
AUDIO_DIR = os.path.join(MUSETALK_DIR, "data/audio")
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

AUDIO_OUTPUT = os.path.join(AUDIO_DIR, "live_audio.wav")
VIDEO_OUTPUT = os.path.join(MUSETALK_DIR, "results/live_output.mp4")

app = FastAPI()

class ChatRequest(BaseModel):
    text: str

print("------------------------------------------------")
print("🚀 INITIALIZING WARM SERVER...")
print("------------------------------------------------")

# Ensure the scripts module can be found
sys.path.append(MUSETALK_DIR)

from scripts.warm_api import RealTimeInference
from omegaconf import OmegaConf

# Load Configuration
inference_config = OmegaConf.load("configs/inference/realtime_live.yaml")
inference_config.preparation = False 
warm_model = RealTimeInference(inference_config)

print("✅ MODEL LOADED! Ready.")

@app.post("/speak")
async def speak(request: ChatRequest):
    print(f"received request: {request.text}")
    
    # 1. CLEANUP: Delete old audio so we NEVER reuse stale files
    if os.path.exists(RAW_AUDIO_SOURCE):
        os.remove(RAW_AUDIO_SOURCE)
    if os.path.exists(AUDIO_OUTPUT):
        os.remove(AUDIO_OUTPUT)

    # 2. GENERATE NEW VOICE
    print("🎤 Generating Voice...")
    try:
        # Run the voice script inside the VoiceEngine directory
        cmd = [VOICE_ENV_PYTHON, VOICE_SCRIPT, request.text]
        subprocess.run(cmd, cwd=VOICE_DIR, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Voice Script Crashed: {e}")
        raise HTTPException(status_code=500, detail="Voice generation crashed")
    except FileNotFoundError:
         print(f"❌ Could not find Voice Script at: {VOICE_DIR}")
         raise HTTPException(status_code=500, detail="VoiceEngine directory not found")

    # 3. VERIFY GENERATION
    if not os.path.exists(RAW_AUDIO_SOURCE):
        print(f"❌ Error: Voice script ran but {RAW_AUDIO_SOURCE} was not created.")
        raise HTTPException(status_code=500, detail="Voice script did not save audio file")

    # 4. CONVERT AUDIO (Move from VoiceEngine -> MuseTalk)
    # We remove stderr=DEVNULL so we can see errors if FFmpeg fails
    result = subprocess.run([
        "ffmpeg", "-y", "-i", RAW_AUDIO_SOURCE, 
        "-ar", "16000", AUDIO_OUTPUT
    ], stdout=subprocess.DEVNULL)

    if result.returncode != 0:
        print("❌ FFmpeg failed to convert audio!")
        raise HTTPException(status_code=500, detail="Audio conversion failed")

    # 5. GENERATE VIDEO
    print("🧠 Generating Video...")
    try:
        output_video_path = warm_model.run(AUDIO_OUTPUT, VIDEO_OUTPUT)
        
        if not output_video_path or not os.path.exists(output_video_path):
             raise Exception("Model returned None or file missing")

    except Exception as e:
        print(f"❌ Video Generation Failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video generation failed: {e}")
    
    return {
        "status": "success", 
        "video_path": output_video_path
    }

@app.get("/download")
async def download_video():
    if os.path.exists(VIDEO_OUTPUT):
        return FileResponse(VIDEO_OUTPUT, media_type="video/mp4", filename="avatar_response.mp4")
    return {"error": "No video generated yet"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import subprocess
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import shutil

# --- CONFIGURATION ---
VOICE_DIR = "/home/ubuntu/VoiceEngine"
VOICE_ENV_PYTHON = os.path.join(VOICE_DIR, "venv/bin/python")
VOICE_SCRIPT = "commercial_voice.py" # Relative to VOICE_DIR
RAW_AUDIO_SOURCE = os.path.join(VOICE_DIR, "output_raw.wav")

MUSETALK_DIR = "/home/ubuntu/MuseTalk"
AUDIO_OUTPUT = os.path.join(MUSETALK_DIR, "data/audio/live_audio.wav")
VIDEO_OUTPUT = "results/live_output.mp4"

app = FastAPI()

class ChatRequest(BaseModel):
    text: str

print("------------------------------------------------")
print("🚀 INITIALIZING WARM SERVER...")
print("------------------------------------------------")

from scripts.warm_api import RealTimeInference
from omegaconf import OmegaConf

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
        # CRITICAL FIX: We set cwd=VOICE_DIR so output_raw.wav creates in the correct folder
        cmd = [VOICE_ENV_PYTHON, VOICE_SCRIPT, request.text]
        subprocess.run(cmd, cwd=VOICE_DIR, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Voice Script Crashed: {e}")
        raise HTTPException(status_code=500, detail="Voice generation crashed")

    # 3. VERIFY GENERATION
    if not os.path.exists(RAW_AUDIO_SOURCE):
        print(f"❌ Error: Voice script ran but {RAW_AUDIO_SOURCE} was not created.")
        raise HTTPException(status_code=500, detail="Voice script did not save audio file")

    # 4. CONVERT AUDIO (Move from VoiceEngine -> MuseTalk)
    subprocess.run([
        "ffmpeg", "-y", "-i", RAW_AUDIO_SOURCE, 
        "-ar", "16000", AUDIO_OUTPUT
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 5. GENERATE VIDEO
    print("🧠 Generating Video...")
    output_video_path = warm_model.run(AUDIO_OUTPUT, VIDEO_OUTPUT)
    
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

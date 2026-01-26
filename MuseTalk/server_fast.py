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
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # .../MuseTalk
MUSETALK_DIR = BASE_DIR
VOICE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../VoiceEngine"))

# Voice Engine Python
if os.path.exists(os.path.join(VOICE_DIR, "venv/bin/python")):
    VOICE_ENV_PYTHON = os.path.join(VOICE_DIR, "venv/bin/python")
else:
    VOICE_ENV_PYTHON = sys.executable

VOICE_SCRIPT = "commercial_voice.py" 
RAW_AUDIO_SOURCE = os.path.join(VOICE_DIR, "output_raw.wav")

# MuseTalk Paths
AUDIO_DIR = os.path.join(MUSETALK_DIR, "data/audio")
if not os.path.exists(AUDIO_DIR): os.makedirs(AUDIO_DIR)

AUDIO_OUTPUT = os.path.join(AUDIO_DIR, "live_audio.wav")
SHORT_CLIP_OUTPUT = os.path.join(MUSETALK_DIR, "results/live_output.mp4")

# --- NEW: CONTINUOUS SESSION PATHS ---
SESSION_OUTPUT = os.path.join(MUSETALK_DIR, "results/full_session.mp4")
CONCAT_LIST_FILE = os.path.join(MUSETALK_DIR, "results/concat_list.txt")

app = FastAPI()

class ChatRequest(BaseModel):
    text: str

print("------------------------------------------------")
print("🚀 INITIALIZING WARM SERVER (CONTINUOUS MODE)...")
print("------------------------------------------------")

sys.path.append(MUSETALK_DIR)
from scripts.warm_api import RealTimeInference
from omegaconf import OmegaConf

# Load Configuration
inference_config = OmegaConf.load("configs/inference/realtime_live.yaml")
inference_config.preparation = False 
warm_model = RealTimeInference(inference_config)

# --- STARTUP CLEANUP ---
# Delete old session on startup so we start fresh
if os.path.exists(SESSION_OUTPUT):
    os.remove(SESSION_OUTPUT)

print("✅ MODEL LOADED! Ready.")

def append_to_session(new_clip_path):
    """
    Appends the new_clip to the SESSION_OUTPUT file using FFmpeg concat.
    """
    # 1. If no session exists yet, just copy the new clip to be the session start
    if not os.path.exists(SESSION_OUTPUT):
        shutil.copy(new_clip_path, SESSION_OUTPUT)
        print("    -> New session started.")
        return

    # 2. If session exists, we must CONCATENATE
    # Create a temporary text file listing the two files to merge
    # FFmpeg requires absolute paths in the list file to be safe
    abs_session = os.path.abspath(SESSION_OUTPUT)
    abs_new = os.path.abspath(new_clip_path)

    with open(CONCAT_LIST_FILE, "w") as f:
        f.write(f"file '{abs_session}'\n")
        f.write(f"file '{abs_new}'\n")

    # Temp file for the merge result
    temp_merge = SESSION_OUTPUT.replace(".mp4", "_temp.mp4")

    # Run FFmpeg Concat (Very fast, no re-encoding)
    # -safe 0 allows absolute paths
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", CONCAT_LIST_FILE, 
        "-c", "copy", # KEY: Copy streams (instant), do not re-encode
        temp_merge
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if result.returncode == 0:
        # Success: Replace old session with new merged file
        os.replace(temp_merge, SESSION_OUTPUT)
        print("    -> Session updated (Appended successfully).")
    else:
        print("    ❌ Error: FFmpeg failed to append video.")

@app.post("/speak")
async def speak(request: ChatRequest):
    print(f"received request: {request.text}")
    
    # 1. CLEANUP OLD TEMP FILES
    if os.path.exists(RAW_AUDIO_SOURCE): os.remove(RAW_AUDIO_SOURCE)
    if os.path.exists(AUDIO_OUTPUT): os.remove(AUDIO_OUTPUT)

    # 2. GENERATE VOICE
    print("🎤 Generating Voice...")
    try:
        cmd = [VOICE_ENV_PYTHON, VOICE_SCRIPT, request.text]
        subprocess.run(cmd, cwd=VOICE_DIR, check=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice Generation Failed: {e}")

    # 3. CONVERT AUDIO
    subprocess.run(["ffmpeg", "-y", "-i", RAW_AUDIO_SOURCE, "-ar", "16000", AUDIO_OUTPUT], stdout=subprocess.DEVNULL)

    # 4. GENERATE VIDEO CLIP
    print("🧠 Generating Video Clip...")
    try:
        # Run warm_api to get the short clip
        short_video_path = warm_model.run(AUDIO_OUTPUT, SHORT_CLIP_OUTPUT)
        
        if not short_video_path or not os.path.exists(short_video_path):
             raise Exception("Model returned None")
        
        # 5. STITCH TO SESSION
        print("🔗 Stitching to Session History...")
        append_to_session(short_video_path)

    except Exception as e:
        print(f"❌ Generation Failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video generation failed: {e}")
    
    # Return the path to the FULL session video
    return {
        "status": "success", 
        "video_path": SESSION_OUTPUT,
        "message": "Video appended to session history."
    }

@app.get("/reset")
async def reset_session():
    """Call this to clear the history and start over."""
    if os.path.exists(SESSION_OUTPUT):
        os.remove(SESSION_OUTPUT)
    return {"status": "Session history cleared."}

@app.get("/download")
async def download_video():
    """Downloads the full accumulated video."""
    if os.path.exists(SESSION_OUTPUT):
        return FileResponse(SESSION_OUTPUT, media_type="video/mp4", filename="full_conversation.mp4")
    return {"error": "No video generated yet"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

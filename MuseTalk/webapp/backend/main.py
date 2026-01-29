# webapp/backend/main.py

import os
import sys
import uuid
import time
import threading
import subprocess
import traceback
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .llm_client import call_llm
from .musetalk_runner import init_musetalk, render_avatar_video


# =========================================================
# Paths (make everything absolute and stable)
# =========================================================
# This file: .../MuseTalk/MuseTalk/webapp/backend/main.py
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))                 # .../webapp/backend
WEBAPP_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))            # .../webapp
MUSETALK_DIR = os.path.abspath(os.path.join(WEBAPP_DIR, ".."))           # .../MuseTalk/MuseTalk  (repo root for MuseTalk code)

FRONTEND_DIR = os.path.abspath(os.path.join(WEBAPP_DIR, "frontend"))     # .../webapp/frontend
OUTPUT_DIR = os.path.join(BACKEND_DIR, "outputs")                        # .../webapp/backend/outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)

# VoiceEngine is usually sibling of MuseTalk/MuseTalk:
# /home/ubuntu/MuseTalk/VoiceEngine
VOICE_DIR = os.path.abspath(os.path.join(MUSETALK_DIR, "..", "VoiceEngine"))
VOICE_SCRIPT = "commercial_voice.py"  # lives inside VOICE_DIR
RAW_AUDIO_SOURCE = os.path.join(VOICE_DIR, "output_raw.wav")  # produced by VoiceEngine script

# Prefer VoiceEngine venv python if it exists
VOICE_ENV_PYTHON = os.path.join(VOICE_DIR, "venv", "bin", "python")
if not os.path.exists(VOICE_ENV_PYTHON):
    VOICE_ENV_PYTHON = sys.executable  # fallback


# =========================================================
# FastAPI App
# =========================================================
app = FastAPI()

# Serve frontend from backend (no separate 5173 server needed)
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
else:
    @app.get("/")
    def missing_frontend():
        return {"error": f"FRONTEND_DIR not found: {FRONTEND_DIR}"}

# CORS (fine for single-user test)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# MuseTalk init (run ONCE at startup)
# =========================================================
class DummyConfig(dict):
    __getattr__ = dict.get

@app.on_event("startup")
def _startup():
    # Ensure we run relative-path dependent code from MUSETALK_DIR
    # so warm_api paths like "data/video/..." work.
    os.chdir(MUSETALK_DIR)

    init_musetalk(DummyConfig({
        "avatar_id": "bank_avatar_1",
        "video_path": "data/video/bank_avatar_25fps.mp4",
    }))


# =========================================================
# Job system
# =========================================================
class ChatRequest(BaseModel):
    user_text: str

jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> dict


def _generate_tts_wav(reply_text: str, out_wav_path: str) -> None:
    """
    Replicates server_fast.py logic:
      - run VoiceEngine script -> creates VOICE_DIR/output_raw.wav
      - ffmpeg convert to 16k wav at out_wav_path
    """
    if not os.path.isdir(VOICE_DIR):
        raise RuntimeError(f"VoiceEngine folder not found: {VOICE_DIR}")

    voice_script_path = os.path.join(VOICE_DIR, VOICE_SCRIPT)
    if not os.path.exists(voice_script_path):
        raise RuntimeError(f"Voice script not found: {voice_script_path}")

    # Cleanup old raw output
    if os.path.exists(RAW_AUDIO_SOURCE):
        os.remove(RAW_AUDIO_SOURCE)
    if os.path.exists(out_wav_path):
        os.remove(out_wav_path)

    # 1) Generate raw voice (creates output_raw.wav)
    cmd = [VOICE_ENV_PYTHON, VOICE_SCRIPT, reply_text]
    subprocess.run(cmd, cwd=VOICE_DIR, check=True)

    if not os.path.exists(RAW_AUDIO_SOURCE):
        raise RuntimeError(f"RAW_AUDIO_SOURCE not produced: {RAW_AUDIO_SOURCE}")

    # 2) Convert to 16k wav (MuseTalk expects 16k)
    # mono helps stability too
    subprocess.run(
        ["ffmpeg", "-y", "-i", RAW_AUDIO_SOURCE, "-ar", "16000", "-ac", "1", out_wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    if not os.path.exists(out_wav_path):
        raise RuntimeError(f"TTS conversion failed, wav not found: {out_wav_path}")


def _run_job(job_id: str, user_text: str):
    try:
        jobs[job_id]["status"] = "llm"
        reply_text = call_llm(user_text)
        jobs[job_id]["reply_text"] = reply_text

        # --- TTS ---
        jobs[job_id]["status"] = "tts"
        wav_path = os.path.join(OUTPUT_DIR, f"{job_id}.wav")
        _generate_tts_wav(reply_text, wav_path)

        # --- Render ---
        jobs[job_id]["status"] = "render"
        out_mp4 = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")

        # Important: ensure relative paths inside warm_api work
        os.chdir(MUSETALK_DIR)

        render_avatar_video(audio_path=wav_path, output_video_path=out_mp4)

        jobs[job_id]["video_path"] = out_mp4
        jobs[job_id]["status"] = "done"

    except Exception:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = traceback.format_exc()


@app.post("/chat")
def chat(req: ChatRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "error": None,
        "video_path": None,
        "reply_text": None,
        "created_at": time.time(),
    }
    threading.Thread(target=_run_job, args=(job_id, req.user_text), daemon=True).start()
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    return {
        "status": job["status"],
        "error": job["error"],
        "reply_text": job["reply_text"],
        "has_video": bool(job["video_path"]),
    }


@app.get("/video/{job_id}")
def video(job_id: str):
    job = jobs.get(job_id)
    if not job or not job["video_path"]:
        raise HTTPException(status_code=404, detail="video_not_ready")
    return FileResponse(job["video_path"], media_type="video/mp4", filename=f"{job_id}.mp4")


@app.get("/health")
def health():
    return {
        "ok": True,
        "musetalk_dir": MUSETALK_DIR,
        "voice_dir": VOICE_DIR,
        "frontend_dir": FRONTEND_DIR,
    }
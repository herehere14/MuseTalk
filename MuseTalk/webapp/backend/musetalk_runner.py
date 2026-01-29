import os, sys

# Add repo root: .../MuseTalk/MuseTalk
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.warm_api import RealTimeInference

# Make a single global model instance so you don't reload every request
_infer = None

def init_musetalk(config):
    global _infer
    if _infer is None:
        _infer = RealTimeInference(config)

def render_avatar_video(audio_path: str, output_video_path: str):
    if _infer is None:
        raise RuntimeError("MuseTalk not initialized. Call init_musetalk(config) at startup.")
    _infer.run(audio_path, output_video_path)
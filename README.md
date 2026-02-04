This is the **Definitive Technical Manual and Documentation** for the **MuseTalk Warm-Server Edition (v1.5 Custom)**. This document serves as a comprehensive guide for developers, researchers, and system architects aiming to deploy real-time digital humans with sub-second latency.

---

Please cd MuseTalk/MuseTalk to start the following commands and install requirements.txt

THe requirements.txt has the latest working dependencies to run this project. However, you also need to download some dependencies through installing github repo and weights. --} use LLM for this please

(you may spend a long time to download and update dependencies)

Also, the mouth movement is still abit unrealistic even after applying multiple algorithms such as Kalman and one-euro algorithm to reduce noise. 



# MuseTalk: Warm-Server Edition (v1.5)

**The Definitive Architecture Guide & Usage Manual**

**Version:** 1.5.0-Warm-Production
**Author:** Customized for Real-Time Digital Human Interaction
**Architecture:** Persistent Latent Inference with Continuous Session Management

---

## 📚 **Table of Contents**

1. **Executive Summary & Core Philosophy**
2. **System Architecture Deep Dive**
* The "Cold Start" Problem
* The "Warm Server" Solution
* The Continuous Session Pipeline


3. **Hardware & Software Prerequisites**
4. **Installation & Environment Setup**
5. **Phase 1: The Force-Prep Protocol (Caching)**
* Theory of Operation
* Execution Guide


6. **Phase 2: High-Fidelity Enhancement**
* GFPGAN Integration
* The Enhancement Workflow


7. **Phase 3: The Warm Server (Runtime)**
* API Endpoints
* Under the Hood: `warm_api.py`
* Stabilization Algorithms (One-Euro, Affine)


8. **Phase 4: The Full-Stack Web Application**
* Backend Job System
* Frontend Polling Architecture


9. **Configuration Reference**
10. **Troubleshooting & Optimization**

---

## 1. Executive Summary & Core Philosophy

Standard implementation of Generative AI video models, specifically **MuseTalk**, suffers from a critical bottleneck known as "Cold Start Latency." In a naive implementation, a request to generate a speaking video triggers a cascade of heavy operations: loading 4GB+ of PyTorch weights into VRAM, instantiating the VAE (Variational Autoencoder) and UNet, reading the source video from disk, running face detection algorithms (like S3FD) on every frame, and finally encoding those frames into latent tensors.

For a 5-second response, this "overhead" can take 10-15 seconds before generation even begins. This renders real-time conversation impossible.

**The MuseTalk Warm-Server Edition** fundamentally re-architects this process into a persistent, stateful service. By decoupling the "Preparation Phase" from the "Inference Phase," we achieve:

1. **Sub-Second Time-to-First-Frame (TTFF):** The model waits in VRAM ("warm"), ready to accept audio tensors instantly.
2. **Zero-Shot Face Detection:** Face coordinates and background latents are pre-calculated and cached (`force_prep.py`), eliminating computer vision overhead during the live chat.
3. **Continuous Conversation:** Instead of generating isolated video files, the server intelligently stitches new sentences into a growing, seamless video file using FFmpeg concat-demuxing (`server_fast.py`).

---

## 2. System Architecture Deep Dive

### The "Cold Start" Problem

In standard inference scripts (like the original `inference.py`), the lifecycle of a request is:

1. `import torch` (2s)
2. `Load UNet/VAE` (4s)
3. `Face Detection` (100ms per frame)
4. `VAE Encoding` (50ms per frame)
5. `UNet Inference` (Fast)
6. `VAE Decoding` (Fast)
7. `Cleanup`

For a 100-frame video (4 seconds), steps 3 and 4 alone add 15 seconds of latency on a mid-range GPU.

Also, when it first render AI generated mouth on top of the video uploaded, the rendering will take up to 10 mins on a low range GPU. But after rendering, we are able to 

### The "Warm Server" Solution

The **Warm Server** (`server_fast.py` + `warm_api.py`) changes the lifecycle to:

1. **Server Start:** Load all models and caches (Done once).
2. **Idle State:** Consume ~6GB VRAM, waiting for requests.
3. **Request Received:**
* Audio → Whisper Feature Extractor (100ms)
* UNet Inference (Direct on GPU)
* VAE Decode & Blend
* **Total Latency:** ~0.8s for start of generation.



### The Continuous Session Pipeline

To simulate a real video call, we cannot just return "video_1.mp4", then "video_2.mp4". We need a single, growing stream.

* **The Session File:** `results/full_session.mp4`
* **The Stitcher:** When a new clip is generated, `server_fast.py` generates a temporary text file list (`concat_list.txt`) containing the absolute paths of the *existing session* and the *new clip*.
* **FFmpeg Concat:** It executes `ffmpeg -f concat -c copy ...`. The `-c copy` flag is crucial; it performs a **bitstream copy**, meaning it does not re-encode frames. It simply updates the container metadata. This allows stitching hours of video in milliseconds.

---

## 3. Hardware & Software Prerequisites

### Hardware

* **GPU:** NVIDIA RTX 3060 (12GB) or higher recommended.
* *Minimum:* 8GB VRAM (e.g., RTX 2070). You may need to reduce batch sizes.
* *Ideal:* RTX 4090 (24GB) allows for batch sizes of 32+ and 4K caching.


* **Storage:** NVMe SSD. Loading large `.pt` latent caches from a spinning HDD will introduce stutter.
* **RAM:** 32GB System RAM.

### Software

* **OS:** Linux (Ubuntu 20.04/22.04) or Windows 10/11 (via PowerShell or CMD).
* **Python:** 3.10.x (Strict dependency).
* **CUDA:** 11.8 or 12.1.
* **FFmpeg:** **CRITICAL**. Must be installed and accessible via system PATH.
* Verify by running `ffmpeg -version` in your terminal.



---

## 4. Installation & Environment Setup

### Step 1: Clone and Environment

```bash
# Clone the repository
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk/MuseTalk

# create and activate environment
python3 venv venv
source venv/bin/activate

# Install PyTorch (Ensure CUDA compatibility)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

```

### Step 2: Install Core Dependencies

Use the provided `requirements.txt`.


```bash
pip install -r requirements.txt
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"

```

*Note: We also need `gfpgan` for the enhancement module.*

```bash
pip install gfpgan basicsr

```

### Step 3: Model Weight Acquisition

You must organize your `models/` directory exactly as follows. Use the `download_weights.sh` script if on Linux, or manually place files for Windows. (Ask AI for this since the dependencies are usually changed frequently)

**Directory Structure:**

```text
./models/
├── musetalkV15/       # The V1.5 UNet Checkpoint
│   ├── musetalk.json
│   └── unet.pth
├── sd-vae/            # Stable Diffusion VAE (ft-mse)
│   ├── config.json
│   └── diffusion_pytorch_model.bin
├── whisper/           # Whisper Tiny (Audio Encoder)
│   ├── config.json
│   └── pytorch_model.bin
├── dwpose/            # Body pose estimation models
├── face-parse-bisent/ # Face Parsing models
├── GFPGANv1.4.pth     # For enhance.py
└── resnet18-5c106cde.pth

```

---

## 5. Phase 1: The Force-Prep Protocol (Caching)

The **Force-Prep** phase is the secret sauce of the Warm Server. It moves the heavy lifting from "Runtime" to "Setup Time."

### Theory of Operation

The script `scripts/force_prep.py` performs the following atomic operations:

1. **Frame Extraction:** Converts the source MP4 into a sequence of PNGs (stored in memory or temp disk).
2. **Face Alignment:** Uses `face_alignment` to detect 68 landmarks. It calculates the bounding box for the face.
3. **Coordinate Caching:** Saves the bounding box coordinates `(x1, y1, x2, y2)` to a pickle file (`coords.pkl`). This ensures the "camera" doesn't jitter during live inference because the coordinates are fixed.
4. **VAE Encoding:** It crops the face, normalizes it to `-1...1`, and passes it through the VAE Encoder. The result is a latent tensor of shape `(4, 32, 32)`.
5. **Latent Caching:** These tensors are concatenated and saved as `latents.pt`.

### Execution Guide

1. **Prepare Source Video:** Place your high-quality avatar video (e.g., `avatar_1.mp4`) in `data/video/`.
3. **Edit Config:** Open `scripts/force_prep.py`.
```python
AVATAR_ID = "my_avatar_v1"
VIDEO_PATH = "data/video/avatar_1.mp4"
BBOX_SHIFT = -5  # Adjusts the chin/mouth crop region

```


3. **Run Script:**
```bash
python -m scripts.force_prep

```


*Result:* A new folder `results/avatars/my_avatar_v1/` is created containing `latents.pt`, `coords.pkl`, etc.

------------------------

However, after this, you may notice you are encountering this error:


no file path: ./models/dwpose/dw-ll_ucoco_384.pth

this is because you haven't downloaded the weight of the open sourced model. the way to fix it is to paste this into terminal:

------------------------------------------------------------------------
python -c '
import os
from huggingface_hub import hf_hub_download

# Define the weights to download
downloads = [
    ("TMElyralab/MuseTalk", "musetalk/musetalk.json", "models/musetalk"),
    ("TMElyralab/MuseTalk", "musetalk/pytorch_model.bin", "models/musetalk"),
    ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", "models/musetalkV15"),
    ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", "models/musetalkV15"),
    ("stabilityai/sd-vae-ft-mse", "config.json", "models/sd-vae"),
    ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", "models/sd-vae"),
    ("openai/whisper-tiny", "config.json", "models/whisper"),
    ("openai/whisper-tiny", "pytorch_model.bin", "models/whisper"),
    ("openai/whisper-tiny", "preprocessor_config.json", "models/whisper"),
    ("yzd-v/DWPose", "dw-ll_ucoco_384.pth", "models/dwpose"),
    ("ByteDance/LatentSync", "latentsync_syncnet.pt", "models/syncnet")
]

print("Starting download...")
for repo, filename, local_dir in downloads:
    print(f"Downloading {filename}...")
    os.makedirs(local_dir, exist_ok=True)
    hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)

# Download ResNet manually (not on HF Hub)
import urllib.request
print("Downloading resnet18...")
os.makedirs("models/face-parse-bisent", exist_ok=True)
urllib.request.urlretrieve(
    "https://download.pytorch.org/models/resnet18-5c106cde.pth", 
    "models/face-parse-bisent/resnet18-5c106cde.pth"
)
print("All downloads complete.")
'
---------------------------------------------------------------------------

If this still returns some other errors
<img width="934" height="338" alt="Screenshot 2026-02-04 at 4 44 48 pm" src="https://github.com/user-attachments/assets/cb7677b2-bee6-4fd1-9ed9-18c0759e5d2f" />

apply this:

---------------------------------------------------------------------------

python -c '
import os
import shutil
from huggingface_hub import hf_hub_download

def fix_or_download(repo, filename, expected_path, correct_local_dir_arg):
    # 1. Check if file exists in the "wrong" nested location
    # Previous script likely made: models/musetalkV15/musetalkV15/unet.pth
    nested_path = os.path.join(os.path.dirname(expected_path), filename)
    
    if os.path.exists(nested_path):
        print(f"Found nested file at {nested_path}. Moving to {expected_path}...")
        os.rename(nested_path, expected_path)
        # Try to remove empty nested dir
        try:
            os.rmdir(os.path.dirname(nested_path))
        except:
            pass
    elif os.path.exists(expected_path):
        print(f"✅ File already exists at {expected_path}")
    else:
        print(f"File missing. Downloading {filename} to {expected_path}...")
        # To get models/musetalkV15/unet.pth, we set local_dir="models" 
        # because filename already contains "musetalkV15/"
        hf_hub_download(repo_id=repo, filename=filename, local_dir=correct_local_dir_arg)

# Fix V1.5 UNet
fix_or_download(
    "TMElyralab/MuseTalk", 
    "musetalkV15/unet.pth", 
    "models/musetalkV15/unet.pth",
    "models" 
)

# Fix V1.5 Config
fix_or_download(
    "TMElyralab/MuseTalk", 
    "musetalkV15/musetalk.json", 
    "models/musetalkV15/musetalk.json",
    "models"
)

print("Fix complete. Running prep...")
'
------------------------------------------------------------------------------------



## 6. Phase 2: High-Fidelity Enhancement

If your source video is 1080p or 4K, the standard MuseTalk output (256x256 face crop) might look soft. Use `enhance.py` to fix this.

### GFPGAN Integration

The `enhance.py` script wraps the **GFPGAN** (Generative Facial Prior GAN) framework. It acts as a restoration filter that "hallucinates" high-frequency details (pores, eyelashes, sharper teeth) onto the generated face.

### The Enhancement Workflow

You have two strategies here:

1. **Pre-Enhancement (Recommended):** Run `enhance.py` on your *source video* before running `force_prep.py`. This ensures the background and non-moving parts of the face are already HD.
2. **Post-Enhancement:** Run `enhance.py` on the final `full_session.mp4` after the chat is over.

**Usage:**
Edit `enhance.py` configuration:

```python
INPUT_VIDEO = "results/full_session.mp4"
OUTPUT_VIDEO = "results/full_session_hd.mp4"

```

Run the script:

```bash
python enhance.py

```

*Note:* The script extracts frames to a `temp_frames` directory, applies `restorer.enhance()`, and then restitches them using FFmpeg to preserve the audio sync.

Before:
<img width="504" height="760" alt="Screenshot 2026-02-04 at 4 46 32 pm" src="https://github.com/user-attachments/assets/bcc47610-78dd-4953-9d42-ff95bb14b478" />

After:
<img width="539" height="950" alt="Screenshot 2026-02-04 at 4 45 33 pm" src="https://github.com/user-attachments/assets/3bdfa7b4-1104-4597-a030-cdf7269415f1" />

---

## 7. Phase 3: The Warm Server (Runtime) (the main file to keep the render stable and effective) IMPORTANT

This is the core of the project. The `server_fast.py` script initializes the `warm_api.py` engine.

### Under the Hood: `warm_api.py`

The `RealTimeInference` class inside `warm_api.py` is a marvel of engineering designed for stability.

#### 1. One-Euro Filtering

Raw face detection is noisy. One frame the box is at `x=100`, the next at `x=101`. This causes the face to shake.
The **One-Euro Filter** (implemented in `OneEuro` class) is a low-pass filter that dynamically adjusts its aggressiveness based on speed.

* **Stationary Face:** High filtering (jitter removal).
* **Moving Head:** Low filtering (low latency tracking).

#### 2. Scale Locking

`warm_api.py` includes a `scale_lock_enabled` flag.

* **Warmup:** For the first 50 frames, it collects width/height statistics.
* **Lock:** It calculates the median size and **locks** the zoom level. This prevents the "pulsing head" effect where the face grows/shrinks slightly as the mouth opens.

#### 3. Affine Tracking & Sub-Pixel Shift

To handle head rotation (roll), the API uses `cv2.calcOpticalFlowPyrLK` (Lucas-Kanade) to track stable features (nose bridge, eyes).
It computes an **Affine Matrix** (`M_affine`) representing the rotation difference between the current frame and the reference. This matrix is applied to the generated mouth patch *before* pasting, ensuring the mouth rotates perfectly with the head.

You can add more features and algorithms in to help with the rendering stability. 

### API Endpoints (`server_fast.py`)

* **`POST /speak`**
* **Payload:** `{"text": "Hello world"}`
* **Action:**
1. Triggers `VoiceEngine` (TTS) to generate `output_raw.wav`.
2. Converts audio to 16kHz via FFmpeg.
3. `warm_model.run()` generates the visual clip.
4. `append_to_session()` stitches the clip to `full_session.mp4`.


* **Return:** JSON with the path to the updated session video.


* **`GET /reset`**
* **Action:** Deletes `full_session.mp4` to start a fresh conversation.



---

## 8. Phase 4: The Full-Stack Web Application (still need to work on, especially the frontend)

Located in `webapp/`, this is a fully functional chat interface.

### Backend Job System (`webapp/backend/main.py`)

Because video generation takes time (even if it's fast), we cannot block the HTTP request.

1. **Request:** User hits `/chat`.
2. **Job Creation:** Backend creates a UUID `job_id` and spawns a **Thread**.
3. **Thread Execution:**
* Call LLM (Simulated or Real).
* Call TTS (`_generate_tts_wav`).
* Call `render_avatar_video`.


4. **Status:** The main thread returns the `job_id` immediately.

### Frontend Polling Architecture (`webapp/frontend/app.js`)

The JavaScript frontend does not wait for a long HTTP response.

1. **Poll:** It checks `/status/{job_id}` every 500ms.
2. **Video Swap:** When status is `done`, it updates the `<video>` src attribute.
3. **Auto-Play Handling:** It manages the `video.play()` promise to avoid browser "User Activation" errors.

---

## 9. Configuration Reference

The behavior of the Warm Server is governed by `configs/inference/realtime_live.yaml`.

| Parameter | Type | Description | Recommended |
| --- | --- | --- | --- |
| `avatar_id` | String | The folder name in `results/avatars/`. Must match `force_prep`. | `bank_avatar_1` |
| `video_path` | Path | Source video file. | `data/video/bank.mp4` |
| `bbox_shift` | Int | Vertical offset for the mouth mask. | `-5` (More chin) |
| `batch_size` | Int | How many frames to process in parallel on GPU. | `4` (RTX 3060), `16` (RTX 4090) |
| `preparation` | Bool | Whether to run face detection at runtime. | **`False`** (Use Cache) |
| `fps` | Int | Target output FPS. | `25` |

### Warm API Internal Toggles

Inside `scripts/warm_api.py`, `RealTimeInference.__init__`:

* `self.blend_mode = 'feather'` or `'poisson'`. Poisson is higher quality but slower (CPU intensive).
* `self.subpixel_enabled = True`. Keeps the face stable.
* `self.scale_lock_enabled = True`. Prevents pulsing.

---

## 10. Troubleshooting & Optimization

### Common Errors

**1. "Error: Input video not found or empty" in `enhance.py**`

* *Cause:* The `INPUT_VIDEO` path in `enhance.py` is incorrect or the video has 0 frames.
* *Fix:* Verify the path logic. Note that `enhance.py` defines `INPUT_VIDEO = "results/..."`. Ensure you are pointing to the correct file.

**2. "FFmpeg command not found"**

* *Cause:* Python's `subprocess` or `os.system` cannot see `ffmpeg`.
* *Fix:* Add FFmpeg to your System PATH variables. Restart your terminal/IDE.

**3. "Model returned None" in `server_fast.py**`

* *Cause:* The inference pipeline failed to generate frames. Usually because the audio file was empty or the `avatar_id` cache is missing.
* *Fix:* Run `force_prep.py` again. Check `logs/` for specific CUDA errors.

**4. Face Jitter / Vibration**

* *Optimization:* In `warm_api.py`, lower the `beta` value in `OneEuro` filter (e.g., `0.05`). This increases smoothing but adds slight latency trails.

**5. OOM (Out of Memory)**

* *Optimization:* Reduce `BATCH_SIZE` in `force_prep.py` and in `realtime_live.yaml`.

import os
import torch
import numpy as np
import pickle
import shutil
from tqdm import tqdm
from musetalk.utils.utils import load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox
import cv2
import glob

# --- CONFIGURATION ---
AVATAR_ID = "bank_avatar_1"
VIDEO_PATH = "data/video/bank_avatar_25fps.mp4"
BBOX_SHIFT = -5
BATCH_SIZE = 4
SAVE_DIR = f"./results/avatars/{AVATAR_ID}"
TEMP_FRAME_DIR = "./results/temp_frames_prep"

def extract_frames_to_disk(video_path, output_dir):
    """
    Extracts frames from MP4 to a folder of PNGs.
    Required because get_landmark_and_bbox expects file paths, not image arrays.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"📹 Extracting frames from {video_path} to {output_dir}...")
    video_stream = cv2.VideoCapture(video_path)
    frame_paths = []
    idx = 0
    
    while True:
        ret, frame = video_stream.read()
        if not ret:
            break
        # Save frame to disk so get_landmark_and_bbox can read it
        filename = os.path.join(output_dir, f"{idx:08d}.png")
        cv2.imwrite(filename, frame)
        frame_paths.append(filename)
        idx += 1
        
    video_stream.release()
    print(f"✅ Extracted {len(frame_paths)} frames.")
    return sorted(frame_paths) # Return list of strings (Paths)

def run_prep():
    print(f"🚀 STARTING FORCE PREP FOR: {AVATAR_ID}")
    
    # 1. Setup Directories
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # 2. Check GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🟢 GPU DETECTED: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("🔴 NO GPU DETECTED. This will be slow!")

    # 3. Load Models
    print("⏳ Loading Models...")
    
    # Unpack exactly 3 items as per your specific version
    try:
        vae, unet, pe = load_all_model()
    except ValueError:
        print("⚠️ Warning: load_all_model returned unexpected number of values. Trying fallback...")
        # Fallback if the version changes again
        results = load_all_model()
        vae = results[0]
        unet = results[1]
        pe = results[2]

    # Move PE to device
    pe.to(device)

    # Move VAE/UNet wrappers to device (try/except just in case they are wrappers)
    try:
        vae.to(device)
        unet.to(device)
    except:
        pass

    # 4. Process Video Frames (Face Detection)
    print("👀 Detecting Face Landmarks...")
    
    # Step A: Extract frames to disk first (Fixes TypeError: expected string)
    frame_paths_list = extract_frames_to_disk(VIDEO_PATH, TEMP_FRAME_DIR)
    
    # Step B: Pass the list of file paths to the detector
    coords_list, frames_list = get_landmark_and_bbox(frame_paths_list, BBOX_SHIFT)
    
    # Save the coordinates
    with open(os.path.join(SAVE_DIR, f"{AVATAR_ID}.pkl"), "wb") as f:
        pickle.dump(coords_list, f)
        
    # 5. Encode Latents (The Heavy Calculation)
    print("🧠 Encoding Video to Latent Space...")
    input_latents_list = []
    input_masks_list = []
    
    # Iterate in batches
    for i in tqdm(range(0, len(frames_list), BATCH_SIZE)):
        batch_frames = frames_list[i : i + BATCH_SIZE]
        batch_coords = coords_list[i : i + BATCH_SIZE]
        
        latents_batch = []
        masks_batch = []
        
        for idx, frame in enumerate(batch_frames):
            x1, y1, x2, y2 = batch_coords[idx]
            
            # Crop Face
            face_img = frame[y1:y2, x1:x2]
            face_img = cv2.resize(face_img, (256, 256))
            
            # Normalize & Move to Device
            face_tensor = torch.tensor(face_img / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Masking
            mask = torch.zeros_like(face_tensor)
            mask[:, :, 128:, :] = 1 # Mask lower half
            
            with torch.no_grad():
                # FIXED: Use .vae.encode to access the inner HuggingFace model
                # The 'vae' object is a MuseTalk wrapper, so we need 'vae.vae'
                latent = vae.vae.encode(face_tensor * 2 - 1).latent_dist.sample() * 0.18215
            
            latents_batch.append(latent)
            masks_batch.append(mask)
            
        input_latents_list.append(torch.cat(latents_batch, dim=0))
        input_masks_list.append(torch.cat(masks_batch, dim=0))

    full_latents = torch.cat(input_latents_list, dim=0)
    full_masks = torch.cat(input_masks_list, dim=0)
    
    # 6. Save Cache
    print("💾 Saving Cache Files...")
    torch.save(full_latents, os.path.join(SAVE_DIR, "latents.pt"))
    torch.save(full_masks, os.path.join(SAVE_DIR, "masks.pt"))
    
    # Create dummy mask coords to satisfy the loader
    dummy_mask_coords = torch.zeros((len(frames_list), 4)) 
    torch.save(dummy_mask_coords, os.path.join(SAVE_DIR, "mask_coords.pt"))
    
    # Clean up temp folder
    shutil.rmtree(TEMP_FRAME_DIR)
    
    print("✅ PREP COMPLETE! You can now run server_fast.py")

if __name__ == "__main__":
    run_prep()

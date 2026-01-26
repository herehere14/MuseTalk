import cv2
import os
import shutil
from gfpgan import GFPGANer

# --- CONFIGURATION ---
INPUT_VIDEO = "results/bank_stable_neg5/v15/bank_avatar_25fps_bank_audio_clean.mp4"
OUTPUT_VIDEO = "results/bank_avatar_hd.mp4"
TEMP_DIR = "temp_frames"

def main():
    # 1. Clean Setup
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    # 2. Load Model
    print("Loading Model...")
    restorer = GFPGANer(
        model_path='models/GFPGANv1.4.pth', 
        upscale=2, 
        arch='clean', 
        channel_multiplier=2, 
        bg_upsampler=None
    )
    
    cap = cv2.VideoCapture(INPUT_VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total == 0:
        print("Error: Input video not found or empty.")
        return

    print(f"Enhancing {total} frames... (Saving as images to avoid codec errors)")
    
    # 3. Process Frame by Frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        # Enhance
        _, _, output = restorer.enhance(
            frame, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True, 
            weight=0.5
        )
        
        # Save as Image (This never fails!)
        cv2.imwrite(f"{TEMP_DIR}/frame_{frame_idx:04d}.jpg", output)
        
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total}")
        frame_idx += 1
        
    cap.release()
    
    # 4. Stitch Images into Video using System FFmpeg (Reliable)
    print("Stitching images back into video...")
    # This command takes the images + original audio -> HD Video
    cmd = (
        f"ffmpeg -y -r {fps} -i {TEMP_DIR}/frame_%04d.jpg -i {INPUT_VIDEO} "
        f"-map 0:v -map 1:a -c:v libx264 -pix_fmt yuv420p -shortest {OUTPUT_VIDEO} -loglevel error"
    )
    os.system(cmd)
    
    # 5. Cleanup
    shutil.rmtree(TEMP_DIR)
    print(f"DONE! Your HD video is ready at: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()

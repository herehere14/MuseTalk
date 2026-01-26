import torch
import os
import numpy as np
import cv2
import subprocess
from tqdm import tqdm

# Import the correct Utils and Classes based on your file structure
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel

class RealTimeInference:
    def __init__(self, config):
        self.avatar_id = config.avatar_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths
        self.cache_dir = os.path.join("results/avatars", self.avatar_id)
        self.latents_path = os.path.join(self.cache_dir, "latents.pt")
        self.coords_path = os.path.join(self.cache_dir, "mask_coords.pt")
        
        # 1. Load Background Video (For Paste-Back)
        self.video_path = config.get("video_path", "data/video/bank_avatar_25fps.mp4")
        print(f"    -> Loading Background Frames from {self.video_path}...")
        self.bg_frames = self.read_video(self.video_path)
        
        # 2. Load Models
        print("    -> Loading VAE, UNet, and PE...")
        loaded_models = load_all_model()
        # Handle the variable return size of load_all_model
        if len(loaded_models) == 4:
            self.vae, self.unet, self.pe = loaded_models[1], loaded_models[2], loaded_models[3]
        elif len(loaded_models) == 3:
            self.vae, self.unet, self.pe = loaded_models[0], loaded_models[1], loaded_models[2]
        
        # Unwrap UNet (The UNet class in unet.py is a wrapper)
        if hasattr(self.unet, 'model'): 
            self.unet_model = self.unet.model
        else: 
            self.unet_model = self.unet

        # Move Models to GPU
        for model in [self.vae, self.unet_model, self.pe]:
            try: model.to(self.device)
            except: pass

        # Detect Precision (Float16 vs Float32)
        try: self.model_dtype = next(self.unet_model.parameters()).dtype
        except: self.model_dtype = torch.float16
        print(f"    -> Model Type: {self.model_dtype}")
        
        # Cast PE & VAE to match Model Type
        if hasattr(self.pe, 'to'): self.pe = self.pe.to(dtype=self.model_dtype, device=self.device)
        if hasattr(self.vae, 'vae'): 
            self.vae.vae = self.vae.vae.to(dtype=self.model_dtype)
        elif hasattr(self.vae, 'to'):
            self.vae = self.vae.to(dtype=self.model_dtype)

        # 3. Initialize Audio Processor (The Real One)
        print("    -> Loading Whisper Audio Processor...")
        self.whisper_dir = "./models/whisper/" 
        self.audio_processor = AudioProcessor(feature_extractor_path=self.whisper_dir)
        # We need the actual Whisper Model to run inference
        self.whisper_model = WhisperModel.from_pretrained(self.whisper_dir).to(self.device, dtype=self.model_dtype).eval()

        # 4. Load Cache
        self.input_latents = torch.load(self.latents_path).to(self.device, dtype=self.model_dtype)
        self.input_coords = torch.load(self.coords_path)
        
        print(f"✅ Avatar '{self.avatar_id}' loaded!")

    def read_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        return frames

    @torch.no_grad()
    def run(self, audio_path, output_video_path):
        # 1. Process Audio
        # Using the AudioProcessor class from musetalk/utils/audio_processor.py
        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
            audio_path, weight_dtype=self.model_dtype
        )
        
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features, 
            self.device, 
            self.model_dtype,
            self.whisper_model, 
            librosa_length, 
            fps=25
        )
        
        # 2. Render Loop
        video_frames = []
        total_frames = whisper_chunks.shape[0]
        cache_len = len(self.input_latents)
        bg_len = len(self.bg_frames)
        
        print(f"    -> Rendering {total_frames} frames...")
        
        for i in tqdm(range(total_frames)):
            frame_idx = i % cache_len
            bg_idx = i % bg_len 
            
            # --- PREPARE INPUTS ---
            # Get the latent from cache. 
            # If it has 8 channels, it's [Masked, Reference]. We want Reference (channels 4-8).
            # If it has 4 channels, it's likely just Reference.
            raw_latent = self.input_latents[frame_idx].unsqueeze(0)
            if raw_latent.shape[1] == 8:
                ref_latent = raw_latent[:, 4:8, :, :] # Extract Reference
            else:
                ref_latent = raw_latent

            # --- FORCE PIXEL MASKING (CRITICAL FOR LIP SYNC) ---
            # 1. Decode to Pixel Space
            if hasattr(self.vae, 'vae'):
                # MuseTalk uses scaling_factor implicitly in encode, so we reverse it here
                pixel_img = self.vae.vae.decode(ref_latent / 0.18215).sample
            else:
                pixel_img = self.vae.decode(ref_latent / 0.18215).sample
            
            # 2. Erase Bottom Half (Set pixels to -1)
            masked_img = pixel_img.clone()
            masked_img[:, :, 128:, :] = -1 
            
            # 3. Encode back to Latent
            if hasattr(self.vae, 'vae'):
                masked_latent = self.vae.vae.encode(masked_img).latent_dist.mode()
            else:
                masked_latent = self.vae.encode(masked_img).latent_dist.mode()
            
            # Re-apply scaling
            masked_latent = masked_latent * 0.18215
            
            # 4. Concatenate [Masked, Reference]
            unet_input = torch.cat([masked_latent, ref_latent], dim=1)
            
            # --- RUN MODEL ---
            # Prepare Audio Embedding
            audio_in = whisper_chunks[i].unsqueeze(0)
            # Apply Positional Encoding to Audio
            audio_with_pe = self.pe(audio_in)
            
            # Run UNet
            pred_latent = self.unet_model(
                sample=unet_input,
                timestep=torch.tensor([0], device=self.device, dtype=self.model_dtype), 
                encoder_hidden_states=audio_with_pe
            ).sample

            # --- DECODE & PASTE ---
            if hasattr(self.vae, 'vae'):
                recon_image = self.vae.vae.decode(pred_latent / 0.18215).sample
            else:
                recon_image = self.vae.decode(pred_latent / 0.18215).sample

            # Post-Process Image
            face_crop = (recon_image[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5
            face_crop = face_crop.cpu().numpy().astype(np.uint8)
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            
            # Paste onto Background
            coords = self.input_coords[frame_idx]
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            full_frame = self.bg_frames[bg_idx].copy()
            
            try:
                face_resized = cv2.resize(face_crop, (x2 - x1, y2 - y1))
                full_frame[y1:y2, x1:x2] = face_resized
            except: pass 

            video_frames.append(full_frame)

        # 3. Save & Merge Audio
        temp_video = output_video_path.replace(".mp4", "_silent.mp4")
        if len(video_frames) > 0:
            h, w, _ = video_frames[0].shape
            out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
            for frame in video_frames:
                out.write(frame)
            out.release()
            
            print("    -> Merging Audio...")
            cmd = f"ffmpeg -y -i {temp_video} -i {audio_path} -c:v copy -c:a aac -strict experimental {output_video_path}"
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(temp_video): os.remove(temp_video)
            
            print(f"    -> Video Saved: {output_video_path}")
            return output_video_path
        return None

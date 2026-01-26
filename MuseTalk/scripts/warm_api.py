import torch
import os
import numpy as np
import cv2
import subprocess
import pickle
import glob
from tqdm import tqdm

# Import Utils
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image_blending
from musetalk.utils.preprocessing import read_imgs
from transformers import WhisperModel

class RealTimeInference:
    def __init__(self, config):
        self.avatar_id = config.avatar_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- PATHS ---
        self.cache_dir = os.path.join("results/v15/avatars", self.avatar_id)
        self.latents_path = os.path.join(self.cache_dir, "latents.pt")
        self.coords_path = os.path.join(self.cache_dir, "coords.pkl")
        self.mask_coords_path = os.path.join(self.cache_dir, "mask_coords.pkl")
        self.mask_out_path = os.path.join(self.cache_dir, "mask")
        
        # 1. Load Background Video
        self.video_path = config.get("video_path", "data/video/bank_avatar_hd.mp4")
        print(f"    -> Loading Background Frames from {self.video_path}...")
        self.bg_frames = self.read_video(self.video_path)
        
        # 2. Load Models
        print("    -> Loading VAE, UNet, and PE...")
        loaded_models = load_all_model()
        if len(loaded_models) == 4:
            self.vae, self.unet, self.pe = loaded_models[1], loaded_models[2], loaded_models[3]
        elif len(loaded_models) == 3:
            self.vae, self.unet, self.pe = loaded_models[0], loaded_models[1], loaded_models[2]
        
        if hasattr(self.unet, 'model'): 
            self.unet_model = self.unet.model
        else: 
            self.unet_model = self.unet

        # Move Models to GPU
        for model in [self.vae, self.unet_model, self.pe]:
            try: model.to(self.device)
            except: pass

        # Detect Precision
        try: self.model_dtype = next(self.unet_model.parameters()).dtype
        except: self.model_dtype = torch.float16
        
        if hasattr(self.pe, 'to'): self.pe = self.pe.to(dtype=self.model_dtype, device=self.device)
        if hasattr(self.vae, 'vae'): 
            self.vae.vae = self.vae.vae.to(dtype=self.model_dtype)
        elif hasattr(self.vae, 'to'):
            self.vae = self.vae.to(dtype=self.model_dtype)

        # 3. Initialize Audio Processor
        print("    -> Loading Whisper Audio Processor...")
        self.whisper_dir = "./models/whisper/" 
        self.audio_processor = AudioProcessor(feature_extractor_path=self.whisper_dir)
        self.whisper_model = WhisperModel.from_pretrained(self.whisper_dir).to(self.device, dtype=self.model_dtype).eval()

        # 4. Load Cache (Latents, Coords, and MASKS)
        print(f"    -> Loading Cache Files...")
        
        # Load Latents
        raw_loaded = torch.load(self.latents_path)
        if isinstance(raw_loaded, list):
            self.input_latents = [t.to(self.device, dtype=self.model_dtype) for t in raw_loaded]
        else:
            self.input_latents = raw_loaded.to(self.device, dtype=self.model_dtype)
        
        # Load Face Coordinates
        with open(self.coords_path, "rb") as f:
            self.input_coords = pickle.load(f)

        # Load Mask Coordinates
        with open(self.mask_coords_path, "rb") as f:
            self.mask_coords_list = pickle.load(f)

        # Load Mask Images (for blending)
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list = read_imgs(input_mask_list)

        print(f"✅ Avatar '{self.avatar_id}' loaded with {len(self.mask_list)} masks!")

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
        
        video_frames = []
        total_frames = whisper_chunks.shape[0]
        cache_len = len(self.input_latents)
        bg_len = len(self.bg_frames)
        
        print(f"    -> Rendering {total_frames} frames...")
        
        for i in tqdm(range(total_frames)):
            frame_idx = i % cache_len
            bg_idx = i % bg_len 
            
            # --- PREPARE LATENTS ---
            raw_latent = self.input_latents[frame_idx]
            if raw_latent.dim() == 3: raw_latent = raw_latent.unsqueeze(0)

            if raw_latent.shape[1] == 8:
                ref_latent = raw_latent[:, 4:8, :, :]
            else:
                ref_latent = raw_latent

            # --- FORCE PIXEL MASKING ---
            if hasattr(self.vae, 'vae'):
                pixel_img = self.vae.vae.decode(ref_latent / 0.18215).sample
            else:
                pixel_img = self.vae.decode(ref_latent / 0.18215).sample
            
            masked_img = pixel_img.clone()
            masked_img[:, :, 128:, :] = -1 
            
            if hasattr(self.vae, 'vae'):
                masked_latent = self.vae.vae.encode(masked_img).latent_dist.mode()
            else:
                masked_latent = self.vae.encode(masked_img).latent_dist.mode()
            
            masked_latent = masked_latent * 0.18215
            unet_input = torch.cat([masked_latent, ref_latent], dim=1)
            
            # --- RUN MODEL ---
            audio_in = whisper_chunks[i].unsqueeze(0)
            audio_with_pe = self.pe(audio_in)
            
            pred_latent = self.unet_model(
                sample=unet_input,
                timestep=torch.tensor([0], device=self.device, dtype=self.model_dtype), 
                encoder_hidden_states=audio_with_pe
            ).sample

            # --- DECODE ---
            if hasattr(self.vae, 'vae'):
                recon_image = self.vae.vae.decode(pred_latent / 0.18215).sample
            else:
                recon_image = self.vae.decode(pred_latent / 0.18215).sample

            # Convert to OpenCV format (BGR)
            face_crop = (recon_image[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5
            face_crop = face_crop.cpu().numpy().astype(np.uint8)
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            
            # --- BLENDING & PASTING (THE FIX) ---
            # 1. Get corresponding background frame
            full_frame = self.bg_frames[bg_idx].copy()
            
            # 2. Get coordinates and masks for this frame
            bbox = self.input_coords[frame_idx % len(self.input_coords)]
            mask = self.mask_list[frame_idx % len(self.mask_list)]
            mask_crop_box = self.mask_coords_list[frame_idx % len(self.mask_coords_list)]

            try:
                # 3. Use MuseTalk's official blending function
                # This handles resizing and soft blending automatically
                final_frame = get_image_blending(
                    full_frame,
                    face_crop,
                    bbox,
                    mask,
                    mask_crop_box
                )
                video_frames.append(final_frame)
            except Exception as e:
                print(f"Blending error at frame {i}: {e}")
                video_frames.append(full_frame) # Fallback to original frame

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
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL) # Allow errors to show if needed
            
            if os.path.exists(temp_video): os.remove(temp_video)
            
            print(f"    -> Video Saved: {output_video_path}")
            return output_video_path
        return None

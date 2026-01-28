import torch
import os
import numpy as np
import cv2
import subprocess
import pickle
import glob
from tqdm import tqdm
from PIL import Image

# Import Utils
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
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
        else:
            raise RuntimeError(f"Unexpected number of models returned by load_all_model(): {len(loaded_models)}")

        if hasattr(self.unet, "model"):
            self.unet_model = self.unet.model
        else:
            self.unet_model = self.unet

        # Move Models to GPU
        for model in [self.vae, self.unet_model, self.pe]:
            try:
                model.to(self.device)
            except Exception:
                pass

        # Detect Precision
        try:
            self.model_dtype = next(self.unet_model.parameters()).dtype
        except Exception:
            self.model_dtype = torch.float16

        if hasattr(self.pe, "to"):
            self.pe = self.pe.to(dtype=self.model_dtype, device=self.device)

        if hasattr(self.vae, "vae"):
            self.vae.vae = self.vae.vae.to(dtype=self.model_dtype)
        elif hasattr(self.vae, "to"):
            self.vae = self.vae.to(dtype=self.model_dtype)

        # 3. Initialize Audio Processor
        print("    -> Loading Whisper Audio Processor...")
        self.whisper_dir = "./models/whisper/"
        self.audio_processor = AudioProcessor(feature_extractor_path=self.whisper_dir)
        self.whisper_model = WhisperModel.from_pretrained(self.whisper_dir).to(
            self.device, dtype=self.model_dtype
        ).eval()

        # 4. Load Cache
        print("    -> Loading Cache Files...")
        raw_loaded = torch.load(self.latents_path)
        if isinstance(raw_loaded, list):
            self.input_latents = [t.to(self.device, dtype=self.model_dtype) for t in raw_loaded]
        else:
            self.input_latents = raw_loaded.to(self.device, dtype=self.model_dtype)

        with open(self.coords_path, "rb") as f:
            self.input_coords = pickle.load(f)

        with open(self.mask_coords_path, "rb") as f:
            self.mask_coords_list = pickle.load(f)

        input_mask_list = glob.glob(os.path.join(self.mask_out_path, "*.[jpJP][pnPN]*[gG]"))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list = read_imgs(input_mask_list)

        print(f"✅ Avatar '{self.avatar_id}' loaded with {len(self.mask_list)} masks!")

    def read_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {path}")
            return []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def safe_blending(self, full_frame, face_img, face_box, mask_img, crop_box):
        """
        A robust blending function that handles edge-of-screen cases explicitly.
        """
        try:
            # Convert to PIL
            body = Image.fromarray(full_frame[:, :, ::-1])  # BGR to RGB
            face = Image.fromarray(face_img[:, :, ::-1])
            if mask_img is not None:
                mask = Image.fromarray(mask_img).convert("L")
            else:
                # Create a default white mask if none provided
                mask = Image.new("L", face.size, 255)

            # Dimensions
            img_w, img_h = body.size
            x, y, x1, y1 = face_box
            cx_s, cy_s, cx_e, cy_e = crop_box  # The larger box used for blending context

            # 1. Calculate the 'Safe' Crop Box (clamped to image bounds)
            safe_cx_s = max(0, cx_s)
            safe_cy_s = max(0, cy_s)
            safe_cx_e = min(img_w, cx_e)
            safe_cy_e = min(img_h, cy_e)

            # If the safe box is invalid (e.g. fully off screen), return original
            if safe_cx_e <= safe_cx_s or safe_cy_e <= safe_cy_s:
                return full_frame

            # 2. Crop the Background (Safe area only)
            bg_crop = body.crop((safe_cx_s, safe_cy_s, safe_cx_e, safe_cy_e))
            bg_w, bg_h = bg_crop.size

            # 3. Build a canvas for crop_box, then crop it down to safe area
            crop_w = cx_e - cx_s
            crop_h = cy_e - cy_s

            # Canvas for Face
            face_canvas = Image.new("RGB", (crop_w, crop_h), (0, 0, 0))
            face_offset_x = x - cx_s
            face_offset_y = y - cy_s
            face_canvas.paste(face, (face_offset_x, face_offset_y))

            # Canvas for Mask
            mask_canvas = Image.new("L", (crop_w, crop_h), 0)
            mask_canvas.paste(mask, (face_offset_x, face_offset_y))

            # 4. Crop the canvases to match the safe background crop
            rel_x = safe_cx_s - cx_s
            rel_y = safe_cy_s - cy_s

            face_patch = face_canvas.crop((rel_x, rel_y, rel_x + bg_w, rel_y + bg_h))
            mask_patch = mask_canvas.crop((rel_x, rel_y, rel_x + bg_w, rel_y + bg_h))

            # 5. Paste safely
            bg_crop.paste(face_patch, (0, 0), mask_patch)

            # 6. Put the patch back into the main image
            body.paste(bg_crop, (safe_cx_s, safe_cy_s))

            return np.array(body)[:, :, ::-1]  # RGB to BGR

        except Exception as e:
            print(f"⚠️ Safe Blending Failed: {e}")
            return full_frame

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
            fps=25,
        )

        video_frames = []
        total_frames = whisper_chunks.shape[0]
        cache_len = len(self.input_latents)
        bg_len = len(self.bg_frames)

        if bg_len == 0:
            print("❌ Error: Background video frames are empty.")
            return None

        print(f"    -> Rendering {total_frames} frames...")

        # Calculate Ratio (e.g. 6050 / 3025 = 2.0)
        frame_ratio = cache_len / bg_len

        for i in tqdm(range(total_frames)):
            # --- FIX: DRIVE EVERYTHING FROM THE BACKGROUND INDEX ---

            # 1. Current step is 'i' (25fps). Find the looping background frame.
            bg_idx = i % bg_len

            # 2. Calculate the EXACT corresponding Cache Frame.
            # If ratio is 2.0:
            # i=0 -> bg=0 -> frame_idx=0
            # i=1 -> bg=1 -> frame_idx=2
            frame_idx = int(bg_idx * frame_ratio)

            # Safety clamp just in case of rounding errors
            if frame_idx >= cache_len:
                frame_idx = cache_len - 1

            # --- PREPARE LATENTS ---
            raw_latent = self.input_latents[frame_idx]
            if raw_latent.dim() == 3:
                raw_latent = raw_latent.unsqueeze(0)

            if raw_latent.shape[1] == 8:
                ref_latent = raw_latent[:, 4:8, :, :]
            else:
                ref_latent = raw_latent

            # --- MASKING ---
            if hasattr(self.vae, "vae"):
                pixel_img = self.vae.vae.decode(ref_latent / 0.18215).sample
            else:
                pixel_img = self.vae.decode(ref_latent / 0.18215).sample

            masked_img = pixel_img.clone()
            masked_img[:, :, 128:, :] = -1

            if hasattr(self.vae, "vae"):
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
                encoder_hidden_states=audio_with_pe,
            ).sample

            # --- DECODE ---
            if hasattr(self.vae, "vae"):
                recon_image = self.vae.vae.decode(pred_latent / 0.18215).sample
            else:
                recon_image = self.vae.decode(pred_latent / 0.18215).sample

            # Convert to OpenCV format (BGR)
            face_crop = (recon_image[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5
            face_crop = face_crop.cpu().numpy().astype(np.uint8)
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

            # --- ROBUST BLENDING ---
            # 1. Get corresponding background frame
            full_frame = self.bg_frames[bg_idx].copy()

            # 2. Get coordinates and masks using bg_idx (VIDEO-synced)
            # Coords/masks were generated per VIDEO frame, not per CACHE frame
            # Using bg_idx ensures stable, non-jittery positioning
            bbox = self.input_coords[bg_idx % len(self.input_coords)]
            mask = self.mask_list[bg_idx % len(self.mask_list)]
            mask_crop_box = self.mask_coords_list[bg_idx % len(self.mask_coords_list)]

            # 3. RESIZE face and mask to fit the target slot
            x, y, x1, y1 = [int(v) for v in bbox]
            target_w = x1 - x
            target_h = y1 - y

            if face_crop.shape[0] != target_h or face_crop.shape[1] != target_w:
                try:
                    face_crop = cv2.resize(face_crop, (target_w, target_h))
                except Exception:
                    pass

            if mask is not None:
                if mask.shape[0] != target_h or mask.shape[1] != target_w:
                    try:
                        mask = cv2.resize(mask, (target_w, target_h))
                    except Exception:
                        pass

            # 4. Call Safe Blending
            final_frame = self.safe_blending(full_frame, face_crop, bbox, mask, mask_crop_box)
            video_frames.append(final_frame)

        # Save & Merge Audio
        temp_video = output_video_path.replace(".mp4", "_silent.mp4")
        if len(video_frames) > 0:
            h, w, _ = video_frames[0].shape
            out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
            for frame in video_frames:
                out.write(frame)
            out.release()

            print("    -> Merging Audio...")
            cmd = (
                f'ffmpeg -y -i "{temp_video}" -i "{audio_path}" '
                f"-c:v copy -c:a aac -strict experimental "
                f'"{output_video_path}"'
            )
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if os.path.exists(temp_video):
                os.remove(temp_video)

            print(f"    -> Video Saved: {output_video_path}")
            return output_video_path

        return None

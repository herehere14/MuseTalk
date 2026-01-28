import torch
import os
import numpy as np
import cv2
import subprocess
import pickle
import glob
import math
from tqdm import tqdm
from PIL import Image

# Import Utils
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.preprocessing import read_imgs
from transformers import WhisperModel


# ==========================================
# ONE-EURO FILTER CLASSES
# ==========================================
def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class LowPass:
    def __init__(self):
        self.inited = False
        self.y = 0.0

    def apply(self, x, a):
        if not self.inited:
            self.inited = True
            self.y = float(x)
            return float(x)
        self.y = a * float(x) + (1.0 - a) * self.y
        return self.y

    def reset(self):
        self.inited = False
        self.y = 0.0


class OneEuro:
    """
    One Euro filter for a scalar.
    Typical tuning:
      min_cutoff ~ 1.0 to 2.0
      beta       ~ 0.03 to 0.20
      d_cutoff   ~ 1.0
    """
    def __init__(self, min_cutoff=1.5, beta=0.08, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_f = LowPass()
        self.dx_f = LowPass()
        self.last_x = None

    def update(self, x, dt):
        x = float(x)
        if self.last_x is None:
            self.last_x = x

        dx = (x - self.last_x) / max(dt, 1e-6)
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = self.dx_f.apply(dx, a_d)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = self.x_f.apply(x, a)

        self.last_x = x
        return x_hat

    def reset(self):
        """Reset filter state for new sequence"""
        self.x_f.reset()
        self.dx_f.reset()
        self.last_x = None

    def snap(self, x):
        """Force filter to snap to a new value (bypass smoothing)"""
        x = float(x)
        self.x_f.inited = True
        self.x_f.y = x
        self.dx_f.reset()
        self.last_x = x
        return x


class RealTimeInference:
    def __init__(self, config):
        self.avatar_id = config.avatar_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==========================================
        # 🔧 TUNING PARAMETERS
        # ==========================================
        self.sync_offset = 0    # Latency adjustment
        self.nose_crop = 0.7    # Safe zone for nose
        self.video_fps = 25     # Output FPS for One-Euro filter dt calculation

        # One-Euro Filter Tuning:
        # - If still jittery: increase min_cutoff (e.g., 2.0) or lower beta
        # - If lagging on head turns: increase beta (e.g., 0.15–0.25)
        self.filter_min_cutoff_pos = 1.5   # For cx, cy (position)
        self.filter_beta_pos = 0.10
        self.filter_min_cutoff_size = 1.0  # For w, h (size)
        self.filter_beta_size = 0.08

        # Snap Gating Tuning:
        # - Higher threshold = less snapping, smoother but may lag
        # - Lower threshold = more snapping, responsive but may jitter
        self.snap_thresh = 12  # pixels - bypass smoothing if delta exceeds this
        self.snap_enabled = True
        # ==========================================

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

        # 2. Load Models (Standard loading...)
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

        for model in [self.vae, self.unet_model, self.pe]:
            try: model.to(self.device)
            except: pass

        # Detect Precision
        try: self.model_dtype = next(self.unet_model.parameters()).dtype
        except: self.model_dtype = torch.float16

        if hasattr(self.pe, "to"): self.pe = self.pe.to(dtype=self.model_dtype, device=self.device)
        if hasattr(self.vae, "vae"): self.vae.vae = self.vae.vae.to(dtype=self.model_dtype)
        elif hasattr(self.vae, "to"): self.vae = self.vae.to(dtype=self.model_dtype)

        # 3. Initialize Audio Processor
        print("    -> Loading Whisper Audio Processor...")
        self.whisper_dir = "./models/whisper/"
        self.audio_processor = AudioProcessor(feature_extractor_path=self.whisper_dir)
        self.whisper_model = WhisperModel.from_pretrained(self.whisper_dir).to(self.device, dtype=self.model_dtype).eval()

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

        # 5. Initialize One-Euro Filters for bbox smoothing
        print("    -> Initializing One-Euro filters with snap gating...")
        self._init_bbox_filters()

        # 6. Build background bbox track using KLT optical flow
        print("    -> Building background bbox track (KLT)...")
        self.bg_bboxes = self._build_bg_bboxes_klt()
        print(f"    -> Built {len(self.bg_bboxes)} bg bboxes")

        print(f"✅ Avatar '{self.avatar_id}' loaded with {len(self.mask_list)} masks!")

    def _init_bbox_filters(self):
        """Initialize One-Euro filters for bounding box coordinates"""
        self.f_cx = OneEuro(min_cutoff=self.filter_min_cutoff_pos, beta=self.filter_beta_pos)
        self.f_cy = OneEuro(min_cutoff=self.filter_min_cutoff_pos, beta=self.filter_beta_pos)
        self.f_w = OneEuro(min_cutoff=self.filter_min_cutoff_size, beta=self.filter_beta_size)
        self.f_h = OneEuro(min_cutoff=self.filter_min_cutoff_size, beta=self.filter_beta_size)

    def _reset_bbox_filters(self):
        """Reset filters for a new rendering session"""
        self.f_cx.reset()
        self.f_cy.reset()
        self.f_w.reset()
        self.f_h.reset()

    def _filter_bbox_with_snap(self, x1, y1, x2, y2):
        """
        Apply One-Euro filter to bounding box with snap gating.
        Snap gating bypasses smoothing when movement exceeds threshold (fast head turns).
        """
        dt = 1.0 / self.video_fps

        # Convert raw bbox to center + size
        cx_raw = (x1 + x2) * 0.5
        cy_raw = (y1 + y2) * 0.5
        w_raw = (x2 - x1)
        h_raw = (y2 - y1)

        # Apply One-Euro filter
        cx_filtered = self.f_cx.update(cx_raw, dt)
        cy_filtered = self.f_cy.update(cy_raw, dt)
        w_filtered = self.f_w.update(w_raw, dt)
        h_filtered = self.f_h.update(h_raw, dt)

        # --- SNAP GATING ---
        # If the raw position moved too far from filtered, snap to raw (fast head turn)
        if self.snap_enabled:
            delta = abs(cx_filtered - cx_raw) + abs(cy_filtered - cy_raw)

            if delta > self.snap_thresh:
                # Bypass smoothing - snap to raw values
                cx_filtered = cx_raw
                cy_filtered = cy_raw
                w_filtered = w_raw
                h_filtered = h_raw

                # Reset filters to the snapped position to avoid oscillation
                self.f_cx.snap(cx_raw)
                self.f_cy.snap(cy_raw)
                self.f_w.snap(w_raw)
                self.f_h.snap(h_raw)

        # Convert back to corner format (PATCH A: use round())
        x1f = int(round(cx_filtered - w_filtered * 0.5))
        y1f = int(round(cy_filtered - h_filtered * 0.5))
        x2f = int(round(cx_filtered + w_filtered * 0.5))
        y2f = int(round(cy_filtered + h_filtered * 0.5))

        return (x1f, y1f, x2f, y2f)

    def _clamp_bbox(self, bbox, W, H):
        """Clamp bbox to image bounds and ensure minimum size."""
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 + 2:
            x2 = min(W - 1, x1 + 3)
        if y2 <= y1 + 2:
            y2 = min(H - 1, y1 + 3)
        return (x1, y1, x2, y2)

    def _build_bg_bboxes_klt(self, max_pts=120, min_pts=30):
        """
        Track bbox across bg_frames using KLT optical flow.
        Returns list of bboxes (len == bg_len) in bg frame coordinates.
        """
        bboxes = []
        bg_len = len(self.bg_frames)
        if bg_len == 0:
            return bboxes

        H, W = self.bg_frames[0].shape[:2]

        # Start bbox from coords that correspond to bg frame 0 time.
        # (Assumes avatar was built from the SAME bg video. If not, rebuild avatar cache.)
        x1, y1, x2, y2 = [int(v) for v in self.input_coords[0]]
        bbox = self._clamp_bbox((x1, y1, x2, y2), W, H)

        prev = cv2.cvtColor(self.bg_frames[0], cv2.COLOR_BGR2GRAY)

        def init_points(gray, bb):
            x1, y1, x2, y2 = bb
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            pts = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=max_pts,
                qualityLevel=0.01,
                minDistance=6,
                blockSize=7,
            )
            if pts is None:
                return None
            pts[:, 0, 0] += x1
            pts[:, 0, 1] += y1
            return pts

        pts = init_points(prev, bbox)
        bboxes.append(bbox)

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        for i in range(1, bg_len):
            curr = cv2.cvtColor(self.bg_frames[i], cv2.COLOR_BGR2GRAY)

            if pts is None or len(pts) < min_pts:
                pts = init_points(prev, bbox)

            if pts is None:
                # fallback: keep last bbox
                bboxes.append(bbox)
                prev = curr
                continue

            nxt, st, err = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None, **lk_params)
            good_old = pts[st == 1]
            good_new = nxt[st == 1]

            if len(good_new) < min_pts:
                bboxes.append(bbox)
                prev = curr
                pts = None
                continue

            # Median motion is robust to outliers
            dx = np.median(good_new[:, 0] - good_old[:, 0])
            dy = np.median(good_new[:, 1] - good_old[:, 1])

            # Shift bbox by median motion
            x1, y1, x2, y2 = bbox
            x1_new = int(round(x1 + dx))
            y1_new = int(round(y1 + dy))
            x2_new = int(round(x2 + dx))
            y2_new = int(round(y2 + dy))

            bbox = self._clamp_bbox((x1_new, y1_new, x2_new, y2_new), W, H)
            bboxes.append(bbox)

            # Update points for next iteration
            pts = good_new.reshape(-1, 1, 2)
            prev = curr

        return bboxes

    # PATCH B: Helper to derive crop box from filtered bbox
    def _crop_box_from_bbox(self, bbox, pad_ratio=0.45):
        """Derive crop box from filtered bbox to ensure alignment."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        pad = int(round(max(w, h) * pad_ratio))  # context margin
        return (x1 - pad, y1 - pad, x2 + pad, y2 + pad)

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
        bg_len = len(self.bg_frames)
        cache_len = len(self.input_latents)

        if bg_len == 0:
            print("❌ Error: Background video frames are empty.")
            return None

        print(f"    -> Rendering {total_frames} frames...")
        print(f"    -> Using KLT-tracked bboxes for stable blending")

        # FIX 1: SPEED SYNC (Ratio Calculation)
        step = max(1, int(round(cache_len / bg_len)))

        # Reset One-Euro filters for this new rendering session (kept for optional use)
        self._reset_bbox_filters()

        for i in tqdm(range(total_frames)):
            # 1. Master Clock: Background Video Loop (25fps)
            bg_idx = i % bg_len

            # 2. Slave Clock: Cache Index (50fps)
            frame_idx = (bg_idx * step) + self.sync_offset

            # 3. Handle Looping / Boundary Safety
            if frame_idx < 0:
                frame_idx = cache_len + frame_idx
            if frame_idx >= cache_len:
                frame_idx = frame_idx % cache_len

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

            face_crop = (recon_image[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5
            face_crop = face_crop.cpu().numpy().astype(np.uint8)
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

            # --- BLENDING ---
            full_frame = self.bg_frames[bg_idx].copy()

            # KLT FIX: Use pre-tracked bbox from background video
            bbox = self.bg_bboxes[bg_idx]
            # Optional: apply additional filtering (usually not needed if KLT is stable)
            # bbox = self._filter_bbox_with_snap(*bbox)

            x, y, x1, y1 = bbox

            mask = self.mask_list[frame_idx]

            # PATCH B: Derive crop box from filtered bbox (guarantees alignment)
            mask_crop_box = self._crop_box_from_bbox(bbox, pad_ratio=0.45)

            # --- FIX 3: RESIZE TO FIT (Dimension Mismatch) ---
            target_w = x1 - x
            target_h = y1 - y

            # Ensure positive dimensions
            if target_w <= 0 or target_h <= 0:
                video_frames.append(full_frame)
                continue

            # Resize Face
            if face_crop.shape[0] != target_h or face_crop.shape[1] != target_w:
                try:
                    face_crop = cv2.resize(face_crop, (target_w, target_h))
                except Exception:
                    pass

            # PATCH C: Proper mask handling (INTER_NEAREST + binarize)
            if mask is not None:
                # If mask is 3-channel, take one channel
                if mask.ndim == 3:
                    mask = mask[:, :, 0]

                # Resize with NEAREST to avoid gray alpha edges
                if mask.shape[0] != target_h or mask.shape[1] != target_w:
                    try:
                        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    except Exception:
                        pass

                # Binarize to 0/255 (critical to prevent blending two mouths)
                mask = (mask > 0).astype(np.uint8) * 255

                # Nose crop (stays binary)
                top_boundary = int(mask.shape[0] * self.nose_crop)
                mask[:top_boundary, :] = 0

                # Zero face pixels based on mask (mask is now guaranteed 2D)
                face_crop[mask == 0] = 0

            # D. Call Safe Blending
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

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
# BLENDING HELPER FUNCTIONS
# ==========================================
def feather_composite(bg_roi, gen_roi, mask_255):
    """
    Feathered boundary blending using OpenCV morphology.
    Hard replace inside, feather only at the boundary.
    """
    m = (mask_255 > 0).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    inner = cv2.erode(m, k3, iterations=1)
    outer = cv2.dilate(m, k9, iterations=1)
    feather = cv2.GaussianBlur(outer, (0, 0), 2.0)
    feather = feather.astype(np.float32) / 255.0
    inner_f = inner.astype(np.float32) / 255.0

    boundary = np.clip(feather - inner_f, 0.0, 1.0)

    out = bg_roi.copy().astype(np.float32)
    gen = gen_roi.astype(np.float32)

    out = out * (1 - inner_f[..., None]) + gen * inner_f[..., None]
    out = out * (1 - boundary[..., None]) + gen * boundary[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)


def poisson_blend(bg_roi, gen_roi, mask_255):
    """
    Poisson (seamless) blending for strongest seam fix.
    """
    if mask_255.sum() < 100:
        return bg_roi

    h, w = bg_roi.shape[:2]
    center = (w // 2, h // 2)

    try:
        blended = cv2.seamlessClone(gen_roi, bg_roi, mask_255, center, cv2.NORMAL_CLONE)
        return blended
    except Exception:
        return feather_composite(bg_roi, gen_roi, mask_255)


def subpixel_shift(img, dx, dy):
    """
    Apply sub-pixel translation using warpAffine.
    """
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return img

    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return shifted


def subpixel_shift_mask(mask, dx, dy):
    """
    Apply sub-pixel translation to mask, then re-binarize.
    """
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return mask

    h, w = mask.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        mask, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return (shifted > 128).astype(np.uint8) * 255


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
        self.x_f.reset()
        self.dx_f.reset()
        self.last_x = None

    def snap(self, x):
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
        self.sync_offset = 0
        self.nose_crop = 0.7
        self.video_fps = 25

        # Position filtering (direct, no motion integration)
        self.filter_min_cutoff_pos = 1.5
        self.filter_beta_pos = 0.10

        # Size filtering (heavy smoothing)
        self.filter_min_cutoff_size = 0.5
        self.filter_beta_size = 0.02

        self.snap_thresh = 12
        self.snap_enabled = True

        # Blending Mode: 'feather' or 'poisson'
        self.blend_mode = 'feather'

        # Sub-pixel alignment
        self.subpixel_enabled = True

        # Scale lock: freeze w/h to median after warmup
        self.scale_lock_enabled = True
        self.scale_lock_warmup_frames = 50

        # Affine transform tracking (micro rotation/scale compensation)
        self.affine_tracking_enabled = True
        self.affine_num_points = 40
        # ==========================================

        # --- PATHS ---
        self.cache_dir = os.path.join("results/v15/avatars", self.avatar_id)
        self.latents_path = os.path.join(self.cache_dir, "latents.pt")
        self.coords_path = os.path.join(self.cache_dir, "coords.pkl")
        self.mask_coords_path = os.path.join(self.cache_dir, "mask_coords.pkl")
        self.mask_out_path = os.path.join(self.cache_dir, "mask")

        self.video_path = config.get("video_path", "data/video/bank_avatar_hd.mp4")
        print(f"    -> Loading Background Frames from {self.video_path}...")
        self.bg_frames = self.read_video(self.video_path)

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

        try: self.model_dtype = next(self.unet_model.parameters()).dtype
        except: self.model_dtype = torch.float16

        if hasattr(self.pe, "to"): self.pe = self.pe.to(dtype=self.model_dtype, device=self.device)
        if hasattr(self.vae, "vae"): self.vae.vae = self.vae.vae.to(dtype=self.model_dtype)
        elif hasattr(self.vae, "to"): self.vae = self.vae.to(dtype=self.model_dtype)

        print("    -> Loading Whisper Audio Processor...")
        self.whisper_dir = "./models/whisper/"
        self.audio_processor = AudioProcessor(feature_extractor_path=self.whisper_dir)
        self.whisper_model = WhisperModel.from_pretrained(self.whisper_dir).to(self.device, dtype=self.model_dtype).eval()

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

        print("    -> Initializing position filters...")
        self._init_filters()

        print("    -> Building background bbox track with stabilization...")
        self.bg_bboxes_float, self.affine_transforms = self._build_bg_bboxes_advanced()
        print(f"    -> Built {len(self.bg_bboxes_float)} bg bboxes (filtered, scale-locked)")

        print(f"✅ Avatar '{self.avatar_id}' loaded with {len(self.mask_list)} masks!")

    def _init_filters(self):
        """Initialize One-Euro filters for direct position filtering"""
        # Position filters (direct filtering, no integration)
        self.f_cx = OneEuro(min_cutoff=self.filter_min_cutoff_pos, beta=self.filter_beta_pos)
        self.f_cy = OneEuro(min_cutoff=self.filter_min_cutoff_pos, beta=self.filter_beta_pos)

        # Size filters (heavy smoothing)
        self.f_w = OneEuro(min_cutoff=self.filter_min_cutoff_size, beta=self.filter_beta_size)
        self.f_h = OneEuro(min_cutoff=self.filter_min_cutoff_size, beta=self.filter_beta_size)

        # State for snap gating
        self.prev_cx_raw = None
        self.prev_cy_raw = None

        # Scale lock state
        self.w_samples = []
        self.h_samples = []
        self.w_locked = None
        self.h_locked = None

    def _reset_filters(self):
        """Reset all filter states"""
        self.f_cx.reset()
        self.f_cy.reset()
        self.f_w.reset()
        self.f_h.reset()

        self.prev_cx_raw = None
        self.prev_cy_raw = None

        self.w_samples = []
        self.h_samples = []
        self.w_locked = None
        self.h_locked = None

    def _filter_position_direct(self, cx_raw, cy_raw, w_raw, h_raw, frame_idx):
        """
        Direct position filtering (no motion integration).
        Anchored to raw KLT tracking each frame - prevents drift.
        """
        dt = 1.0 / self.video_fps

        # Initialize on first frame
        if self.prev_cx_raw is None:
            self.prev_cx_raw = cx_raw
            self.prev_cy_raw = cy_raw

        # Snap gating for large movements
        dx_raw = cx_raw - self.prev_cx_raw
        dy_raw = cy_raw - self.prev_cy_raw

        if self.snap_enabled and (abs(dx_raw) > self.snap_thresh or abs(dy_raw) > self.snap_thresh):
            self.f_cx.snap(cx_raw)
            self.f_cy.snap(cy_raw)
            cx_filtered = cx_raw
            cy_filtered = cy_raw
        else:
            # Direct position filtering (anchored to raw each frame)
            cx_filtered = self.f_cx.update(cx_raw, dt)
            cy_filtered = self.f_cy.update(cy_raw, dt)

        self.prev_cx_raw = cx_raw
        self.prev_cy_raw = cy_raw

        # --- SCALE HANDLING ---
        if self.scale_lock_enabled:
            if frame_idx < self.scale_lock_warmup_frames:
                self.w_samples.append(w_raw)
                self.h_samples.append(h_raw)
                w_final = self.f_w.update(w_raw, dt)
                h_final = self.f_h.update(h_raw, dt)
            else:
                if self.w_locked is None:
                    self.w_locked = float(np.median(self.w_samples))
                    self.h_locked = float(np.median(self.h_samples))
                    print(f"    -> Scale locked: w={self.w_locked:.1f}, h={self.h_locked:.1f}")
                w_final = self.w_locked
                h_final = self.h_locked
        else:
            w_final = self.f_w.update(w_raw, dt)
            h_final = self.f_h.update(h_raw, dt)

        return (cx_filtered, cy_filtered, w_final, h_final)

    def _clamp_bbox(self, bbox, W, H):
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

    def _build_bg_bboxes_advanced(self, max_pts=120, min_pts=30):
        """
        Build bbox track with:
        1. KLT optical flow for raw tracking
        2. Direct position filtering (no drift)
        3. Scale locking after warmup
        4. Affine transform estimation for micro rotation/scale
        """
        bboxes_float = []
        affine_transforms = []
        bg_len = len(self.bg_frames)

        if bg_len == 0:
            return bboxes_float, affine_transforms

        H, W = self.bg_frames[0].shape[:2]

        # Initial bbox from coords
        x1, y1, x2, y2 = [float(v) for v in self.input_coords[0]]
        cx_raw = (x1 + x2) * 0.5
        cy_raw = (y1 + y2) * 0.5
        w_raw = x2 - x1
        h_raw = y2 - y1

        prev_gray = cv2.cvtColor(self.bg_frames[0], cv2.COLOR_BGR2GRAY)

        def init_affine_points(gray, cx, cy, w, h, num_pts):
            """Initialize tracking points in nose/cheek region (stable texture)"""
            x1i = max(0, int(cx - w * 0.3))
            y1i = max(0, int(cy - h * 0.2))
            x2i = min(W, int(cx + w * 0.3))
            y2i = min(H, int(cy + h * 0.4))
            roi = gray[y1i:y2i, x1i:x2i]
            if roi.size == 0:
                return None
            pts = cv2.goodFeaturesToTrack(
                roi, maxCorners=num_pts, qualityLevel=0.01, minDistance=5, blockSize=7,
            )
            if pts is None:
                return None
            pts[:, 0, 0] += x1i
            pts[:, 0, 1] += y1i
            return pts

        def init_bbox_points(gray, cx, cy, w, h):
            x1i = max(0, int(cx - w * 0.5))
            y1i = max(0, int(cy - h * 0.5))
            x2i = min(W, int(cx + w * 0.5))
            y2i = min(H, int(cy + h * 0.5))
            roi = gray[y1i:y2i, x1i:x2i]
            if roi.size == 0:
                return None
            pts = cv2.goodFeaturesToTrack(
                roi, maxCorners=max_pts, qualityLevel=0.01, minDistance=6, blockSize=7,
            )
            if pts is None:
                return None
            pts[:, 0, 0] += x1i
            pts[:, 0, 1] += y1i
            return pts

        # Initialize tracking points
        pts_affine = init_affine_points(prev_gray, cx_raw, cy_raw, w_raw, h_raw, self.affine_num_points)
        pts_bbox = init_bbox_points(prev_gray, cx_raw, cy_raw, w_raw, h_raw)

        # Reset filters
        self._reset_filters()

        # First frame
        cx_f, cy_f, w_f, h_f = self._filter_position_direct(cx_raw, cy_raw, w_raw, h_raw, 0)
        bboxes_float.append((cx_f, cy_f, w_f, h_f))
        affine_transforms.append(np.eye(2, 3, dtype=np.float32))

        lk_params = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        for i in range(1, bg_len):
            curr_gray = cv2.cvtColor(self.bg_frames[i], cv2.COLOR_BGR2GRAY)

            # --- AFFINE TRANSFORM ESTIMATION ---
            M_affine = np.eye(2, 3, dtype=np.float32)

            if self.affine_tracking_enabled and pts_affine is not None and len(pts_affine) >= 4:
                nxt_affine, st_affine, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, pts_affine, None, **lk_params
                )
                good_old_affine = pts_affine[st_affine == 1]
                good_new_affine = nxt_affine[st_affine == 1]

                if len(good_new_affine) >= 4:
                    M, inliers = cv2.estimateAffinePartial2D(
                        good_old_affine, good_new_affine, method=cv2.RANSAC
                    )
                    if M is not None:
                        M_affine = M.astype(np.float32)
                    pts_affine = good_new_affine.reshape(-1, 1, 2)
                else:
                    pts_affine = init_affine_points(curr_gray, cx_raw, cy_raw, w_raw, h_raw, self.affine_num_points)
            else:
                pts_affine = init_affine_points(curr_gray, cx_raw, cy_raw, w_raw, h_raw, self.affine_num_points)

            affine_transforms.append(M_affine)

            # --- BBOX TRACKING ---
            if pts_bbox is None or len(pts_bbox) < min_pts:
                pts_bbox = init_bbox_points(prev_gray, cx_raw, cy_raw, w_raw, h_raw)

            if pts_bbox is None:
                cx_f, cy_f, w_f, h_f = self._filter_position_direct(cx_raw, cy_raw, w_raw, h_raw, i)
                bboxes_float.append((cx_f, cy_f, w_f, h_f))
                prev_gray = curr_gray
                continue

            nxt_bbox, st_bbox, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_bbox, None, **lk_params)
            good_old_bbox = pts_bbox[st_bbox == 1]
            good_new_bbox = nxt_bbox[st_bbox == 1]

            if len(good_new_bbox) < min_pts:
                cx_f, cy_f, w_f, h_f = self._filter_position_direct(cx_raw, cy_raw, w_raw, h_raw, i)
                bboxes_float.append((cx_f, cy_f, w_f, h_f))
                prev_gray = curr_gray
                pts_bbox = None
                continue

            # Median motion
            dx = float(np.median(good_new_bbox[:, 0] - good_old_bbox[:, 0]))
            dy = float(np.median(good_new_bbox[:, 1] - good_old_bbox[:, 1]))

            # Update raw center
            cx_raw = cx_raw + dx
            cy_raw = cy_raw + dy

            # Clamp to frame bounds
            cx_raw = max(w_raw * 0.5, min(W - w_raw * 0.5, cx_raw))
            cy_raw = max(h_raw * 0.5, min(H - h_raw * 0.5, cy_raw))

            # Apply direct position filtering + scale lock
            cx_f, cy_f, w_f, h_f = self._filter_position_direct(cx_raw, cy_raw, w_raw, h_raw, i)
            bboxes_float.append((cx_f, cy_f, w_f, h_f))

            pts_bbox = good_new_bbox.reshape(-1, 1, 2)
            prev_gray = curr_gray

        return bboxes_float, affine_transforms

    def _float_bbox_to_int(self, cx, cy, w, h):
        """
        Convert float (cx, cy, w, h) to int bbox.
        Returns both int bbox and sub-pixel offsets.
        """
        fx = cx - w * 0.5
        fy = cy - h * 0.5

        ix = int(math.floor(fx))
        iy = int(math.floor(fy))

        dx = fx - ix
        dy = fy - iy

        x1 = ix
        y1 = iy
        x2 = int(round(cx + w * 0.5))
        y2 = int(round(cy + h * 0.5))

        return (x1, y1, x2, y2), (dx, dy)

    def _crop_box_from_bbox(self, bbox, pad_ratio=0.45):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        pad = int(round(max(w, h) * pad_ratio))
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

    def _apply_affine_to_patch(self, face_img, mask_img, M_affine, target_size):
        """
        Apply affine transform for micro rotation/scale compensation.
        """
        if not self.affine_tracking_enabled:
            return face_img, mask_img

        identity = np.eye(2, 3, dtype=np.float32)
        diff = np.abs(M_affine - identity).max()
        if diff < 1e-4:
            return face_img, mask_img

        h, w = face_img.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        a, b = M_affine[0, 0], M_affine[0, 1]
        c, d = M_affine[1, 0], M_affine[1, 1]

        M_centered = np.float32([
            [a, b, cx * (1 - a) - cy * b],
            [c, d, cy * (1 - d) - cx * c]
        ])

        face_warped = cv2.warpAffine(
            face_img, M_centered, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        mask_warped = mask_img
        if mask_img is not None:
            mask_warped = cv2.warpAffine(
                mask_img, M_centered, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            mask_warped = (mask_warped > 128).astype(np.uint8) * 255

        return face_warped, mask_warped

    def advanced_blending(self, full_frame, face_img, face_box, mask_img, crop_box,
                          subpixel_offset=(0.0, 0.0), affine_matrix=None):
        """
        Advanced blending with feather/Poisson, sub-pixel alignment, and affine compensation.
        """
        try:
            img_h, img_w = full_frame.shape[:2]
            x, y, x1, y1 = face_box
            cx_s, cy_s, cx_e, cy_e = crop_box
            dx, dy = subpixel_offset

            if x1 <= x or y1 <= y:
                return full_frame

            safe_cx_s = max(0, cx_s)
            safe_cy_s = max(0, cy_s)
            safe_cx_e = min(img_w, cx_e)
            safe_cy_e = min(img_h, cy_e)

            if safe_cx_e <= safe_cx_s or safe_cy_e <= safe_cy_s:
                return full_frame

            bg_roi = full_frame[safe_cy_s:safe_cy_e, safe_cx_s:safe_cx_e].copy()

            crop_w = cx_e - cx_s
            crop_h = cy_e - cy_s

            face_canvas = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
            rel_x_bg = safe_cx_s - cx_s
            rel_y_bg = safe_cy_s - cy_s
            face_canvas[rel_y_bg:rel_y_bg + bg_roi.shape[0],
                        rel_x_bg:rel_x_bg + bg_roi.shape[1]] = bg_roi.copy()

            face_offset_x = x - cx_s
            face_offset_y = y - cy_s
            fh, fw = face_img.shape[:2]

            # Apply affine transform
            if affine_matrix is not None:
                face_img, mask_img = self._apply_affine_to_patch(
                    face_img, mask_img, affine_matrix, (fw, fh)
                )

            # Apply sub-pixel shift
            if self.subpixel_enabled and (abs(dx) > 1e-4 or abs(dy) > 1e-4):
                face_img = subpixel_shift(face_img, dx, dy)
                if mask_img is not None:
                    mask_img = subpixel_shift_mask(mask_img, dx, dy)

            fx_start = max(0, face_offset_x)
            fy_start = max(0, face_offset_y)
            fx_end = min(crop_w, face_offset_x + fw)
            fy_end = min(crop_h, face_offset_y + fh)

            src_x_start = fx_start - face_offset_x
            src_y_start = fy_start - face_offset_y
            src_x_end = src_x_start + (fx_end - fx_start)
            src_y_end = src_y_start + (fy_end - fy_start)

            if fx_end > fx_start and fy_end > fy_start:
                face_canvas[fy_start:fy_end, fx_start:fx_end] = \
                    face_img[src_y_start:src_y_end, src_x_start:src_x_end]

            mask_canvas = np.zeros((crop_h, crop_w), dtype=np.uint8)
            if mask_img is not None:
                if mask_img.ndim == 3:
                    mask_img = mask_img[:, :, 0]
                mh, mw = mask_img.shape[:2]

                mx_start = max(0, face_offset_x)
                my_start = max(0, face_offset_y)
                mx_end = min(crop_w, face_offset_x + mw)
                my_end = min(crop_h, face_offset_y + mh)

                msrc_x_start = mx_start - face_offset_x
                msrc_y_start = my_start - face_offset_y
                msrc_x_end = msrc_x_start + (mx_end - mx_start)
                msrc_y_end = msrc_y_start + (my_end - my_start)

                if mx_end > mx_start and my_end > my_start:
                    mask_canvas[my_start:my_end, mx_start:mx_end] = \
                        mask_img[msrc_y_start:msrc_y_end, msrc_x_start:msrc_x_end]

            rel_x = safe_cx_s - cx_s
            rel_y = safe_cy_s - cy_s
            bg_h, bg_w = bg_roi.shape[:2]

            gen_roi = face_canvas[rel_y:rel_y + bg_h, rel_x:rel_x + bg_w]
            mask_roi = mask_canvas[rel_y:rel_y + bg_h, rel_x:rel_x + bg_w]

            mask_255 = (mask_roi > 0).astype(np.uint8) * 255

            if self.blend_mode == 'poisson':
                blended_roi = poisson_blend(bg_roi, gen_roi, mask_255)
            else:
                blended_roi = feather_composite(bg_roi, gen_roi, mask_255)

            result = full_frame.copy()
            result[safe_cy_s:safe_cy_e, safe_cx_s:safe_cx_e] = blended_roi

            return result

        except Exception as e:
            print(f"⚠️ Advanced Blending Failed: {e}")
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
        print(f"    -> Direct position filtering + scale-lock + affine + {self.blend_mode} blending")

        step = max(1, int(round(cache_len / bg_len)))

        for i in tqdm(range(total_frames)):
            bg_idx = i % bg_len
            frame_idx = (bg_idx * step) + self.sync_offset

            if frame_idx < 0:
                frame_idx = cache_len + frame_idx
            if frame_idx >= cache_len:
                frame_idx = frame_idx % cache_len

            raw_latent = self.input_latents[frame_idx]
            if raw_latent.dim() == 3:
                raw_latent = raw_latent.unsqueeze(0)

            if raw_latent.shape[1] == 8:
                ref_latent = raw_latent[:, 4:8, :, :]
            else:
                ref_latent = raw_latent

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

            audio_in = whisper_chunks[i].unsqueeze(0)
            audio_with_pe = self.pe(audio_in)

            pred_latent = self.unet_model(
                sample=unet_input,
                timestep=torch.tensor([0], device=self.device, dtype=self.model_dtype),
                encoder_hidden_states=audio_with_pe,
            ).sample

            if hasattr(self.vae, "vae"):
                recon_image = self.vae.vae.decode(pred_latent / 0.18215).sample
            else:
                recon_image = self.vae.decode(pred_latent / 0.18215).sample

            face_crop = (recon_image[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5
            face_crop = face_crop.cpu().numpy().astype(np.uint8)
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

            full_frame = self.bg_frames[bg_idx].copy()

            # Get float bbox and affine transform
            cx, cy, w, h = self.bg_bboxes_float[bg_idx]
            bbox, subpixel_offset = self._float_bbox_to_int(cx, cy, w, h)
            affine_matrix = self.affine_transforms[bg_idx] if self.affine_tracking_enabled else None

            x, y, x1, y1 = bbox

            mask = self.mask_list[frame_idx]
            mask_crop_box = self._crop_box_from_bbox(bbox, pad_ratio=0.45)

            target_w = x1 - x
            target_h = y1 - y

            if target_w <= 0 or target_h <= 0:
                video_frames.append(full_frame)
                continue

            if face_crop.shape[0] != target_h or face_crop.shape[1] != target_w:
                try:
                    face_crop = cv2.resize(face_crop, (target_w, target_h))
                except Exception:
                    pass

            if mask is not None:
                if mask.ndim == 3:
                    mask = mask[:, :, 0]

                if mask.shape[0] != target_h or mask.shape[1] != target_w:
                    try:
                        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    except Exception:
                        pass

                mask = (mask > 0).astype(np.uint8) * 255

                top_boundary = int(mask.shape[0] * self.nose_crop)
                mask[:top_boundary, :] = 0

            final_frame = self.advanced_blending(
                full_frame, face_crop, bbox, mask, mask_crop_box,
                subpixel_offset=subpixel_offset,
                affine_matrix=affine_matrix
            )
            video_frames.append(final_frame)

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

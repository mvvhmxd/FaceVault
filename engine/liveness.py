import numpy as np
import cv2
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HAS_MP = False
_landmarker = None

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )

    HAS_MP = True
except ImportError:
    pass


def _get_landmarker():
    global _landmarker
    if _landmarker is not None:
        return _landmarker
    if not HAS_MP:
        return None
    model_path = Path("data/face_landmarker.task")
    if not model_path.exists():
        return None
    try:
        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
        )
        _landmarker = FaceLandmarker.create_from_options(opts)
    except Exception as e:
        logger.warning(f"Liveness FaceLandmarker init failed: {e}")
    return _landmarker


class _FaceTrack:
    """Per-face temporal state keyed by approximate bbox position."""
    def __init__(self):
        self.blink_history: list[bool] = []
        self.motion_history: list[float] = []
        self.prev_gray = None
        self.frame_count = 0
        self.ever_blinked = False


class LivenessDetector:
    def __init__(self):
        self._tracks: dict[str, _FaceTrack] = {}
        self.frame_count = 0

    def _track_key(self, bbox):
        """Quantize bbox center into grid cells so nearby positions share state."""
        if bbox is None:
            return "default"
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return f"{cx // 80}_{cy // 80}"

    def _get_track(self, bbox) -> _FaceTrack:
        key = self._track_key(bbox)
        if key not in self._tracks:
            self._tracks[key] = _FaceTrack()
        # Evict stale tracks
        if len(self._tracks) > 10:
            oldest = min(self._tracks, key=lambda k: self._tracks[k].frame_count)
            if oldest != key:
                del self._tracks[oldest]
        return self._tracks[key]

    def check(self, image, bbox=None):
        face = self._crop(image, bbox)
        if face is None or face.size == 0:
            return {"is_live": False, "score": 0.0, "checks": {}}

        track = self._get_track(bbox)

        checks = {
            "texture": self._texture(face),
            "moire": self._moire(face),
            "reflection": self._reflection(face),
            "color": self._color(face),
            "blink": self._blink(face, track),
            "motion": self._motion(face, track),
            "frequency": self._frequency(face),
            "context": self._context(image, bbox),
            "size": self._size_check(image, bbox),
        }

        track.frame_count += 1
        self.frame_count += 1
        fc = track.frame_count

        # ── Scoring ──
        w = {
            "texture": 0.06,
            "moire": 0.10,
            "reflection": 0.06,
            "color": 0.05,
            "blink": 0.25,
            "motion": 0.13,
            "frequency": 0.10,
            "context": 0.15,
            "size": 0.10,
        }
        score = sum(checks[k] * w[k] for k in checks)

        # ── Boost: confirmed blinks are proof of life ──
        if track.ever_blinked:
            score = max(score, 0.62)
        if checks["blink"] > 0.85:
            score = max(score, 0.70)
        if checks["motion"] > 0.8 and checks["blink"] > 0.6:
            score += 0.08

        # ── Veto: no blinks over time = photo/screen ──
        if fc > 15 and not track.ever_blinked:
            score = min(score, 0.45)
        if fc > 30 and not track.ever_blinked:
            score = min(score, 0.35)

        # ── Veto: strong screen signals ──
        if checks["context"] < 0.2:
            score = min(score, 0.30)
        if checks["moire"] < 0.2 and checks["reflection"] < 0.2:
            score = min(score, 0.30)

        score = max(0.0, min(1.0, score))

        return {
            "is_live": score > 0.50,
            "score": round(score, 3),
            "checks": {k: round(v, 3) for k, v in checks.items()},
        }

    def _crop(self, image, bbox):
        if bbox is None:
            return image
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = image.shape[:2]
        c = image[max(0, y1): min(h, y2), max(0, x1): min(w, x2)]
        return c if c.size > 0 else None

    # ── Texture: LBP variance to detect flat / printed surfaces ──

    def _texture(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        lbp = self._compute_lbp(gray)
        lbp_var = float(np.var(lbp))

        if lap_var < 8:
            return 0.15
        if lbp_var < 300:
            return 0.3

        score = min(1.0, (lap_var / 80) * 0.5 + (lbp_var / 2000) * 0.5)
        return max(0.2, min(1.0, score))

    def _compute_lbp(self, gray):
        h, w = gray.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j]   >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1]   >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j]   >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1]   >= center) << 0
                lbp[i-1, j-1] = code
        return lbp

    # ── Moiré pattern detection (screens produce periodic pixel grids) ──

    def _moire(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128)).astype(np.float64)

        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.log(np.abs(fshift) + 1)

        h, w = mag.shape
        cy, cx = h // 2, w // 2

        # Mask out DC component (center 5x5)
        mag_masked = mag.copy()
        mag_masked[cy-2:cy+3, cx-2:cx+3] = 0

        # High-frequency ring (outer 25% of spectrum)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        r_max = min(cx, cy)
        hf_mask = dist > r_max * 0.6
        hf_energy = float(np.mean(mag_masked[hf_mask]))

        # Mid-frequency energy
        mf_mask = (dist > r_max * 0.2) & (dist <= r_max * 0.6)
        mf_energy = float(np.mean(mag_masked[mf_mask]))

        # Screens have strong periodic peaks in high-frequency band
        hf_peaks = mag_masked[hf_mask]
        peak_ratio = float(np.max(hf_peaks) / (np.mean(hf_peaks) + 1e-6))

        # Also check for horizontal/vertical line artifacts (screen refresh)
        h_line = float(np.mean(mag_masked[cy, :]))
        v_line = float(np.mean(mag_masked[:, cx]))
        line_strength = (h_line + v_line) / (np.mean(mag_masked) + 1e-6)

        # High peak ratio or strong line artifacts = likely screen
        if peak_ratio > 8.0 or line_strength > 3.0:
            return 0.15
        if peak_ratio > 5.0 or line_strength > 2.0:
            return 0.3
        if hf_energy > mf_energy * 0.8:
            return 0.35

        return 0.8

    # ── Screen reflection / glare detection ──

    def _reflection(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        hsv = cv2.resize(hsv, (64, 64))

        # Screens have specular highlights (very bright, low saturation spots)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]

        bright_mask = v_channel > 220
        bright_ratio = float(np.sum(bright_mask) / bright_mask.size)

        # Bright + low saturation = screen glare
        if bright_ratio > 0:
            bright_sat = float(np.mean(s_channel[bright_mask]))
        else:
            bright_sat = 100

        # Uniformity of brightness (screens have even backlight)
        block_size = 16
        blocks = []
        for i in range(0, 64, block_size):
            for j in range(0, 64, block_size):
                blocks.append(float(np.mean(gray[i:i+block_size, j:j+block_size])))

        brightness_uniformity = 1.0 - (float(np.std(blocks)) / (float(np.mean(blocks)) + 1e-6))

        score = 0.75
        if bright_ratio > 0.25 and bright_sat < 30:
            score -= 0.35
        if brightness_uniformity > 0.90:
            score -= 0.25
        if bright_ratio > 0.4:
            score -= 0.15

        b_channel = face[:, :, 0] if len(face.shape) == 3 else gray
        b_small = cv2.resize(b_channel, (64, 64))
        unique_ratio = len(np.unique(b_small)) / 256.0
        if unique_ratio < 0.15:
            score -= 0.2

        return max(0.05, min(1.0, score))

    # ── Color distribution ──

    def _color(self, face):
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        hsv = cv2.resize(hsv, (64, 64))

        # Check skin-tone presence
        mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 200, 255]))
        skin = float(np.sum(mask > 0) / mask.size)

        # Saturation statistics
        sat = hsv[:, :, 1].astype(np.float64)
        sat_std = float(np.std(sat))
        sat_mean = float(np.mean(sat))

        # Real skin has varied saturation; screens tend toward uniform
        score = 0.5
        if 0.2 < skin < 0.8:
            score += 0.15
        if sat_std > 25:
            score += 0.2
        if sat_mean < 15 or sat_mean > 180:
            score -= 0.2  # unnatural saturation

        # Check blue channel dominance (screens emit more blue)
        b, g, r = cv2.split(cv2.resize(face, (64, 64)))
        blue_ratio = float(np.mean(b)) / (float(np.mean(r)) + 1e-6)
        if blue_ratio > 1.3:
            score -= 0.2  # screen blue shift

        return max(0.1, min(1.0, score))

    # ── Blink detection (requires temporal history) ──

    def _blink(self, face, track: _FaceTrack):
        landmarker = _get_landmarker()
        if landmarker is None:
            return 0.3

        try:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)

            if not result.face_blendshapes or not result.face_blendshapes[0]:
                return 0.2

            bs = {b.category_name: b.score for b in result.face_blendshapes[0]}
            blink_l = bs.get("eyeBlinkLeft", 0)
            blink_r = bs.get("eyeBlinkRight", 0)
            avg_blink = (blink_l + blink_r) / 2
            is_blink = avg_blink > 0.35

            if is_blink:
                track.ever_blinked = True

            track.blink_history.append(is_blink)
            if len(track.blink_history) > 60:
                track.blink_history = track.blink_history[-60:]

            total = len(track.blink_history)
            blink_count = sum(track.blink_history)

            if total < 8:
                return 0.5

            if track.ever_blinked:
                blink_rate = blink_count / total
                if 0.01 < blink_rate < 0.35:
                    return 0.95
                return 0.7

            # Never blinked
            if total > 30:
                return 0.1
            if total > 15:
                return 0.25
            return 0.4
        except Exception:
            return 0.3

    # ── Motion consistency ──

    def _motion(self, face, track: _FaceTrack):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64)).astype(np.float32)
        score = 0.4

        if track.prev_gray is not None:
            diff = float(np.mean(np.abs(gray - track.prev_gray)))
            track.motion_history.append(diff)
            if len(track.motion_history) > 30:
                track.motion_history = track.motion_history[-30:]

            if len(track.motion_history) > 8:
                mv = float(np.var(track.motion_history))
                mean_motion = float(np.mean(track.motion_history))

                if mv > 1.0 and mean_motion > 0.3:
                    score = 0.9
                elif mv < 0.05:
                    score = 0.15
                elif mean_motion < 0.1 and len(track.motion_history) > 15:
                    score = 0.2
                else:
                    score = 0.55

        track.prev_gray = gray
        return score

    # ── Frequency analysis ──

    def _frequency(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
        h, w = mag.shape

        center = mag[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        outer_ring = np.concatenate([
            mag[:h//4, :].flatten(),
            mag[3*h//4:, :].flatten(),
            mag[:, :w//4].flatten(),
            mag[:, 3*w//4:].flatten(),
        ])

        center_energy = float(np.mean(center))
        outer_energy = float(np.mean(outer_ring))
        ratio = center_energy / (outer_energy + 1e-6)

        # Check for periodic spikes (screen pixel grid)
        outer_std = float(np.std(outer_ring))
        outer_max = float(np.max(outer_ring))
        spike_ratio = outer_max / (float(np.mean(outer_ring)) + 1e-6)

        if spike_ratio > 4.0:
            return 0.2  # strong periodic pattern = screen
        if ratio > 6.0:
            return 0.3  # too much low-frequency = blurred/screen
        if 1.5 < ratio < 5.0 and spike_ratio < 3.0:
            return 0.8

        return 0.4

    # ── Face size relative to frame (phone faces are small) ──

    def _size_check(self, image, bbox):
        if bbox is None:
            return 0.7
        x1, y1, x2, y2 = (int(v) for v in bbox)
        ih, iw = image.shape[:2]
        face_area = (x2 - x1) * (y2 - y1)
        frame_area = ih * iw
        ratio = face_area / (frame_area + 1e-6)

        # A real face at normal webcam distance covers 5-40% of the frame
        # A phone-screen face covers ~1-5%
        if ratio < 0.02:
            return 0.1   # tiny face = almost certainly a screen
        if ratio < 0.04:
            return 0.25
        if ratio < 0.07:
            return 0.45
        if ratio > 0.40:
            return 0.6   # very close, unusual but not spoof
        return 0.85

    # ── Context: analyze area AROUND the face for screen borders ──

    def _context(self, image, bbox):
        if bbox is None:
            return 0.7
        x1, y1, x2, y2 = (int(v) for v in bbox)
        ih, iw = image.shape[:2]
        fw, fh = x2 - x1, y2 - y1

        # Expand bbox by 60% on each side to get surrounding context
        pad_x = int(fw * 0.6)
        pad_y = int(fh * 0.6)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(iw, x2 + pad_x)
        cy2 = min(ih, y2 + pad_y)

        context = image[cy1:cy2, cx1:cx2]
        if context.size == 0:
            return 0.5

        gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (96, 96))

        score = 0.8

        # 1) Detect strong straight edges around face (phone/screen border)
        edges = cv2.Canny(gray, 50, 150)
        # Look at border regions of the context (outer 20%)
        bw = 96 // 5
        border_mask = np.zeros_like(edges)
        border_mask[:bw, :] = 1
        border_mask[-bw:, :] = 1
        border_mask[:, :bw] = 1
        border_mask[:, -bw:] = 1
        # Exclude the face center
        fc = 96 // 4
        border_mask[fc:96-fc, fc:96-fc] = 0

        border_edges = float(np.sum(edges * border_mask)) / (np.sum(border_mask) + 1e-6)

        if border_edges > 40:
            score -= 0.3  # strong edges around face = likely phone border
        elif border_edges > 25:
            score -= 0.15

        # 2) Check for sharp brightness contrast between face area and surround
        face_region = gray[96//4:3*96//4, 96//4:3*96//4]
        surround_pixels = np.concatenate([
            gray[:bw, :].flatten(),
            gray[-bw:, :].flatten(),
            gray[:, :bw].flatten(),
            gray[:, -bw:].flatten(),
        ])
        face_bright = float(np.mean(face_region))
        surr_bright = float(np.mean(surround_pixels))
        contrast = abs(face_bright - surr_bright)

        if contrast > 60:
            score -= 0.25  # screen face is brighter/darker than surroundings
        elif contrast > 40:
            score -= 0.1

        # 3) Check color temperature difference (screen vs natural light)
        ctx_color = cv2.resize(context, (32, 32))
        face_color = ctx_color[8:24, 8:24]
        surr_color = np.concatenate([
            ctx_color[:6, :].reshape(-1, 3),
            ctx_color[-6:, :].reshape(-1, 3),
            ctx_color[:, :6].reshape(-1, 3),
            ctx_color[:, -6:].reshape(-1, 3),
        ])

        face_b = float(np.mean(face_color[:, :, 0]))
        face_r = float(np.mean(face_color[:, :, 2]))
        surr_b = float(np.mean(surr_color[:, 0]))
        surr_r = float(np.mean(surr_color[:, 2]))

        # Screen faces have different blue/red ratio vs surrounding
        face_temp = face_b / (face_r + 1e-6)
        surr_temp = surr_b / (surr_r + 1e-6)
        temp_diff = abs(face_temp - surr_temp)

        if temp_diff > 0.3:
            score -= 0.2  # very different color temperature
        elif temp_diff > 0.15:
            score -= 0.1

        return max(0.05, min(1.0, score))

    def reset(self):
        self._tracks.clear()
        self.frame_count = 0

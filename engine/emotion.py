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
    logger.warning("mediapipe not available — emotion uses fallback")


def _get_landmarker():
    global _landmarker
    if _landmarker is not None:
        return _landmarker
    if not HAS_MP:
        return None
    model_path = Path("data/face_landmarker.task")
    if not model_path.exists():
        logger.warning("face_landmarker.task not found — emotion uses fallback")
        return None
    try:
        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
        )
        _landmarker = FaceLandmarker.create_from_options(opts)
        logger.info("MediaPipe FaceLandmarker ready")
    except Exception as e:
        logger.warning(f"FaceLandmarker init failed: {e}")
    return _landmarker


class EmotionDetector:
    EMOTIONS = ["neutral", "happy", "sad", "surprise", "angry", "fear", "disgust"]

    def detect(self, image, bbox=None):
        face_img = self._crop(image, bbox)
        if face_img is None or face_img.size == 0:
            return {"emotion": "neutral", "confidence": 0.5, "scores": {}}

        landmarker = _get_landmarker()

        if landmarker is not None:
            result = self._detect_with_blendshapes(face_img, landmarker)
            if result:
                return result

        return self._detect_with_geometry(face_img, landmarker)

    def _crop(self, image, bbox):
        if bbox is None:
            return image
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = image.shape[:2]
        c = image[max(0, y1): min(h, y2), max(0, x1): min(w, x2)]
        return c if c.size > 0 else None

    def _detect_with_blendshapes(self, face_img, landmarker):
        """Use MediaPipe blendshapes for emotion estimation."""
        try:
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)

            if not result.face_landmarks:
                return None

            lm = result.face_landmarks[0]

            bs = {}
            if result.face_blendshapes and len(result.face_blendshapes) > 0:
                for b in result.face_blendshapes[0]:
                    bs[b.category_name] = b.score

            return self._score_from_blendshapes(bs, lm)
        except Exception as e:
            logger.debug(f"Blendshape detection failed: {e}")
            return None

    def _score_from_blendshapes(self, bs, lm):
        s = {}

        smile = bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)
        jaw_open = bs.get("jawOpen", 0)
        brow_down = bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)
        brow_up = bs.get("browInnerUp", 0) + bs.get("browOuterUpLeft", 0) + bs.get("browOuterUpRight", 0)
        eye_wide = bs.get("eyeWideLeft", 0) + bs.get("eyeWideRight", 0)
        eye_squint = bs.get("eyeSquintLeft", 0) + bs.get("eyeSquintRight", 0)
        mouth_frown = bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)
        nose_sneer = bs.get("noseSneerLeft", 0) + bs.get("noseSneerRight", 0)

        s["happy"] = min(1.0, smile * 0.6 + eye_squint * 0.2)
        s["surprise"] = min(1.0, jaw_open * 0.4 + eye_wide * 0.4 + brow_up * 0.3)
        s["sad"] = min(1.0, mouth_frown * 0.5 + brow_up * 0.15)
        s["angry"] = min(1.0, brow_down * 0.5 + nose_sneer * 0.3)
        s["fear"] = min(1.0, eye_wide * 0.4 + brow_up * 0.2 + jaw_open * 0.15)
        s["disgust"] = min(1.0, nose_sneer * 0.5 + mouth_frown * 0.2)

        other_max = max(s.values()) if s else 0
        s["neutral"] = max(0.0, 1.0 - other_max * 1.5)

        total = sum(s.values()) + 1e-6
        s = {k: v / total for k, v in s.items()}
        top = max(s, key=s.get)

        return {
            "emotion": top,
            "confidence": round(s[top], 3),
            "scores": {k: round(v, 3) for k, v in sorted(s.items(), key=lambda x: -x[1])},
        }

    def _detect_with_geometry(self, face_img, landmarker):
        """Fallback: simple geometry-based emotion estimation."""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        top_half = gray[: h // 2, :]
        bottom_half = gray[h // 2:, :]
        top_var = float(np.var(top_half))
        bot_var = float(np.var(bottom_half))
        ratio = bot_var / (top_var + 1e-6)

        s = {
            "happy": min(1.0, max(0, ratio - 0.8) * 2),
            "surprise": min(1.0, max(0, top_var / 2000)),
            "sad": min(1.0, max(0, 1.2 - ratio) * 0.5),
            "angry": min(1.0, max(0, top_var / 3000)),
            "neutral": 0.4,
            "fear": 0.05,
            "disgust": 0.03,
        }

        total = sum(s.values()) + 1e-6
        s = {k: v / total for k, v in s.items()}
        top = max(s, key=s.get)

        return {
            "emotion": top,
            "confidence": round(s[top], 3),
            "scores": {k: round(v, 3) for k, v in sorted(s.items(), key=lambda x: -x[1])},
        }

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

_face_app = None
_initialized = False


def _get_app():
    global _face_app, _initialized
    if _initialized:
        return _face_app
    try:
        from insightface.app import FaceAnalysis

        _face_app = FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"]
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
        _initialized = True
        logger.info("FaceProcessor initialized with buffalo_l")
    except Exception as e:
        logger.warning(f"buffalo_l unavailable ({e}), trying buffalo_sc")
        try:
            from insightface.app import FaceAnalysis

            _face_app = FaceAnalysis(
                name="buffalo_sc", providers=["CPUExecutionProvider"]
            )
            _face_app.prepare(ctx_id=0, det_size=(320, 320))
            _initialized = True
            logger.info("FaceProcessor initialized with buffalo_sc")
        except Exception as e2:
            logger.error(f"InsightFace initialization failed: {e2}")
            _initialized = True  # prevent retry loops
    return _face_app


class FaceProcessor:
    def __init__(self):
        self.target_size = 640

    def initialize(self):
        _get_app()

    @property
    def ready(self):
        return _get_app() is not None

    def process_frame(self, image):
        app = _get_app()
        if app is None:
            return []

        h, w = image.shape[:2]
        scale = min(self.target_size / max(w, h), 1.0)
        if scale < 1.0:
            resized = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            resized = image
            scale = 1.0

        faces = app.get(resized)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            if scale < 1.0:
                bbox = (bbox / scale).astype(int)

            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            results.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "embedding": face.embedding.astype(np.float32),
                    "age": int(face.age),
                    "gender": "Male" if face.sex == "M" else "Female",
                    "det_score": round(float(face.det_score), 3),
                    "landmarks": face.kps.tolist() if face.kps is not None else None,
                }
            )

        return results

    def extract_embedding(self, image):
        results = self.process_frame(image)
        if results:
            return results[0]["embedding"]
        return None

    def get_face_crop(self, image, bbox, size=96):
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        pad = int((x2 - x1) * 0.15)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (size, size))

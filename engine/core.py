import numpy as np
import cv2
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FaceResult:
    __slots__ = ('bbox', 'embedding', 'age', 'gender', 'confidence', 'keypoints')

    def __init__(self, bbox, embedding, age, gender, confidence, keypoints=None):
        self.bbox = bbox
        self.embedding = embedding
        self.age = age
        self.gender = gender
        self.confidence = confidence
        self.keypoints = keypoints

    def to_dict(self):
        return {
            'bbox': self.bbox,
            'embedding': self.embedding,
            'age': self.age,
            'gender': self.gender,
            'confidence': self.confidence,
            'keypoints': self.keypoints,
        }


class FaceEngine:
    MODEL_PRIORITY = ['buffalo_l', 'buffalo_sc', 'buffalo_s']

    def __init__(self, det_size=(640, 640)):
        self._app = None
        self._det_size = det_size
        self._initialize()

    def _initialize(self):
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            logger.error("insightface not installed")
            return

        for model_name in self.MODEL_PRIORITY:
            try:
                self._app = FaceAnalysis(
                    name=model_name,
                    providers=['CPUExecutionProvider'],
                )
                self._app.prepare(ctx_id=0, det_size=self._det_size)
                logger.info("FaceEngine loaded model: %s", model_name)
                return
            except Exception:
                continue

        logger.error("No InsightFace model could be loaded")

    @property
    def ready(self) -> bool:
        return self._app is not None

    def detect_faces(self, image: np.ndarray, max_dim: int = 640, min_dim: int = 300) -> List[FaceResult]:
        if not self.ready:
            return []

        h, w = image.shape[:2]
        longest = max(h, w)

        pad = 0
        if longest < min_dim:
            pad = int(longest * 0.4)
            padded = np.zeros((h + pad * 2, w + pad * 2, 3), dtype=image.dtype)
            padded[pad:pad + h, pad:pad + w] = image
            scale = min_dim / max(padded.shape[:2])
            proc = cv2.resize(padded, (int(padded.shape[1] * scale),
                                       int(padded.shape[0] * scale)),
                              interpolation=cv2.INTER_LINEAR)
        elif longest > max_dim:
            scale = max_dim / longest
            proc = cv2.resize(image, (int(w * scale), int(h * scale)))
        else:
            proc = image
            scale = 1.0

        faces = self._app.get(proc)
        results = []

        for face in faces:
            bbox = face.bbox.astype(float)
            if scale != 1.0:
                bbox /= scale
            if pad > 0:
                bbox[0] -= pad; bbox[1] -= pad
                bbox[2] -= pad; bbox[3] -= pad

            kps = None
            if face.kps is not None:
                kps = face.kps.astype(float)
                if scale != 1.0:
                    kps /= scale
                if pad > 0:
                    kps -= pad
                kps = kps.tolist()

            gender_raw = getattr(face, 'gender', None) or getattr(face, 'sex', None)
            if isinstance(gender_raw, str):
                gender = 'Male' if gender_raw.upper().startswith('M') else 'Female'
            elif isinstance(gender_raw, (int, float)):
                gender = 'Male' if int(gender_raw) == 1 else 'Female'
            else:
                gender = None

            results.append(FaceResult(
                bbox=[int(x) for x in bbox],
                embedding=face.embedding.tolist() if face.embedding is not None else None,
                age=int(face.age) if hasattr(face, 'age') and face.age is not None else None,
                gender=gender,
                confidence=float(face.det_score),
                keypoints=kps,
            ))

        return results

    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        faces = self.detect_faces(image)
        if not faces or faces[0].embedding is None:
            return None
        return np.array(faces[0].embedding)

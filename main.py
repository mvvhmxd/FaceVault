import base64
import time
import logging
import json
import os
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from engine.database import FaceDatabase
from engine.face_processor import FaceProcessor
from engine.emotion import EmotionDetector
from engine.liveness import LivenessDetector
from engine.celebrity import CelebrityMatcher
from engine.reconstruction import ReconstructionEngine
from engine.privacy import PrivacyAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("facevault")

LITE_MODE = os.environ.get("LITE_MODE", "").lower() in ("1", "true", "yes")

db = FaceDatabase()
processor = FaceProcessor()
emotion_det = None if LITE_MODE else EmotionDetector()
liveness_det = None if LITE_MODE else LivenessDetector()
celebrity = CelebrityMatcher()
recon_engine = ReconstructionEngine(db)
privacy_analyzer = PrivacyAnalyzer(recon_engine)

if LITE_MODE:
    logging.getLogger("facevault").info("Running in LITE mode (no emotion/liveness)")

_models_ready = False
_model_error = None


def _warmup():
    global _models_ready, _model_error
    try:
        logger.info("Background model loading started...")
        processor.initialize()
        if processor.ready:
            logger.info("Models loaded and ready")
            _models_ready = True
        else:
            _model_error = "Model initialized but not ready"
            logger.error(_model_error)
            _models_ready = True  # let requests through so errors are visible
    except Exception as e:
        _model_error = str(e)
        logger.error(f"Model loading FAILED: {e}")
        _models_ready = True  # unblock requests so health endpoint reports error


@asynccontextmanager
async def lifespan(application: FastAPI):
    thread = threading.Thread(target=_warmup, daemon=True)
    thread.start()
    yield


app = FastAPI(
    title="FaceVault — Real-Time Face Recognition System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/")
async def root():
    return FileResponse("static/index.html")


def _decode_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    return img


def _decode_b64(data: str) -> np.ndarray:
    if not data or len(data) < 20:
        raise HTTPException(400, "Invalid or empty image data")
    try:
        if "," in data:
            data = data.split(",", 1)[1]
        raw = base64.b64decode(data)
    except Exception:
        raise HTTPException(400, "Invalid base64 encoding")
    return _decode_image(raw)


def _sanitize_name(name: str) -> str:
    import re
    name = name.strip()
    name = re.sub(r"[<>&\"'/\\;]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name[:50]


# ── Main real-time endpoint ─────────────────────────────────────────
@app.post("/api/process-frame")
async def process_frame(image: str = Form(...)):
    if not _models_ready:
        return {"faces": [], "count": 0, "processing_ms": 0, "loading": True}
    t0 = time.perf_counter()
    img = _decode_b64(image)

    # Downscale for memory on constrained environments
    h, w = img.shape[:2]
    if max(h, w) > 480:
        scale = 480 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    faces = processor.process_frame(img)

    results = []
    for f in faces:
        emb = f["embedding"]
        bbox = f["bbox"]

        identity = db.find_match(emb, threshold=0.4)

        if emotion_det:
            emo = emotion_det.detect(img, bbox)
        else:
            emo = {"emotion": "neutral", "confidence": 0.5, "scores": {}}

        if liveness_det:
            live = liveness_det.check(img, bbox)
        else:
            live = {"is_live": True, "score": 0.6, "checks": {}}

        celeb = celebrity.match(emb, top_k=1)

        results.append({
            "bbox": bbox,
            "name": identity["name"] if identity else "Unknown",
            "confidence": round(identity["score"], 3) if identity else 0.0,
            "age": f["age"],
            "gender": f["gender"],
            "det_score": f["det_score"],
            "emotion": emo["emotion"],
            "emotion_confidence": emo["confidence"],
            "emotion_scores": emo["scores"],
            "is_live": live["is_live"],
            "liveness_score": live["score"],
            "liveness_checks": live["checks"],
            "celebrity_match": celeb[0] if celeb else None,
        })

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    return {"faces": results, "count": len(results), "processing_ms": elapsed}


# ── Face registration ────────────────────────────────────────────────
@app.post("/api/register")
async def register_face(name: str = Form(...), image: str = Form(...)):
    name = _sanitize_name(name)
    if not name:
        raise HTTPException(400, "Name cannot be empty")
    if len(name) < 2:
        raise HTTPException(400, "Name must be at least 2 characters")

    img = _decode_b64(image)
    faces = processor.process_frame(img)
    if not faces:
        raise HTTPException(400, "No face detected — make sure your face is clearly visible")

    # Pick the largest face if multiple detected
    if len(faces) > 1:
        faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)

    f = faces[0]
    emb = f["embedding"]

    # Check for duplicate face (same person already registered)
    dup = db.find_duplicate_face(emb, threshold=0.65)
    if dup:
        # Same face exists — update if same name, warn if different name
        if dup["name"].lower() == name.lower():
            thumb_bytes = None
            crop = processor.get_face_crop(img, f["bbox"])
            if crop is not None:
                _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                thumb_bytes = buf.tobytes()
            db.update_face(dup["id"], emb, thumb_bytes)
            return {
                "id": dup["id"],
                "name": name,
                "updated": True,
                "message": f"Updated embedding for '{name}' (similarity {dup['score']:.0%})",
            }
        else:
            raise HTTPException(
                409,
                f"This face is already registered as '{dup['name']}' "
                f"(similarity {dup['score']:.0%}). Remove it first to re-register under a new name.",
            )

    # Check for duplicate name
    if db.name_exists(name):
        raise HTTPException(
            409,
            f"The name '{name}' is already registered. Use a different name or remove the existing entry first.",
        )

    thumb_bytes = None
    crop = processor.get_face_crop(img, f["bbox"])
    if crop is not None:
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        thumb_bytes = buf.tobytes()

    fid = db.add_face(
        name,
        emb,
        thumbnail_bytes=thumb_bytes,
        metadata={"age": f["age"], "gender": f["gender"]},
    )
    multi_warn = f" ({len(faces)} faces detected, registered the largest)" if len(faces) > 1 else ""
    return {"id": fid, "name": name, "message": f"Registered '{name}' successfully{multi_warn}"}


# ── Embedding endpoints ─────────────────────────────────────────────
@app.post("/api/extract-embedding")
async def extract_embedding(image: str = Form(...)):
    img = _decode_b64(image)
    faces = processor.process_frame(img)
    if not faces:
        raise HTTPException(400, "No face detected")
    emb = faces[0]["embedding"]
    return {
        "embedding": emb.tolist(),
        "dimensions": len(emb),
        "age": faces[0]["age"],
        "gender": faces[0]["gender"],
    }


@app.post("/api/compare-embeddings")
async def compare_embeddings(embedding1: str = Form(...), embedding2: str = Form(...)):
    try:
        e1 = np.array(json.loads(embedding1), dtype=np.float32)
        e2 = np.array(json.loads(embedding2), dtype=np.float32)
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(400, "Invalid embedding format — expected JSON array of numbers")
    if e1.ndim != 1 or e2.ndim != 1:
        raise HTTPException(400, "Embeddings must be 1-dimensional arrays")
    if len(e1) != len(e2):
        raise HTTPException(400, f"Embedding dimension mismatch: {len(e1)} vs {len(e2)}")
    n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if n1 < 1e-6 or n2 < 1e-6:
        raise HTTPException(400, "Embedding is a zero vector")
    sim = float(np.dot(e1, e2) / (n1 * n2))
    return {
        "similarity": round(sim, 4),
        "is_same_person": sim > 0.4,
        "confidence": round(abs(sim - 0.4) / 0.6 * 100, 1),
    }


# ── Reconstruction ───────────────────────────────────────────────────
@app.post("/api/reconstruct-face")
async def reconstruct_face(image: str = Form(...), method: str = Form("nearest_blend")):
    img = _decode_b64(image)
    faces = processor.process_frame(img)
    if not faces:
        raise HTTPException(400, "No face detected")
    emb = faces[0]["embedding"]
    result = recon_engine.reconstruct(emb, method=method)
    return result


# ── Privacy score ────────────────────────────────────────────────────
@app.post("/api/privacy-score")
async def privacy_score(image: str = Form(...)):
    img = _decode_b64(image)
    faces = processor.process_frame(img)
    if not faces:
        raise HTTPException(400, "No face detected")
    return privacy_analyzer.analyze(faces[0]["embedding"])


# ── Spoof check ──────────────────────────────────────────────────────
@app.post("/api/spoof-check")
async def spoof_check(image: str = Form(...)):
    img = _decode_b64(image)
    faces = processor.process_frame(img)
    if not faces:
        raise HTTPException(400, "No face detected")

    live = liveness_det.check(img, faces[0]["bbox"])

    emb = faces[0]["embedding"]
    emb_consistency = 1.0
    match = db.find_match(emb, threshold=0.3)
    if match:
        stored = db.get_all_embeddings().get(match["id"])
        if stored is not None:
            emb_consistency = float(
                np.dot(emb, stored) / (np.linalg.norm(emb) * np.linalg.norm(stored) + 1e-6)
            )

    is_spoof = not live["is_live"] or emb_consistency < 0.5
    return {
        "is_spoof": is_spoof,
        "liveness": live,
        "embedding_consistency": round(emb_consistency, 3),
        "verdict": "SPOOF DETECTED" if is_spoof else "GENUINE",
    }


# ── Celebrity match ──────────────────────────────────────────────────
@app.post("/api/celebrity-match")
async def celebrity_match(image: str = Form(...)):
    img = _decode_b64(image)
    faces = processor.process_frame(img)
    if not faces:
        raise HTTPException(400, "No face detected")
    matches = celebrity.match(faces[0]["embedding"], top_k=5)
    return {"matches": matches}


# ── Interpolation ────────────────────────────────────────────────────
@app.post("/api/interpolate")
async def interpolate(image1: str = Form(...), image2: str = Form(...), steps: int = Form(5)):
    e1 = processor.extract_embedding(_decode_b64(image1))
    e2 = processor.extract_embedding(_decode_b64(image2))
    if e1 is None or e2 is None:
        raise HTTPException(400, "Face not detected in one or both images")
    frames = recon_engine.interpolate(e1, e2, steps=steps)
    return {"frames": frames}


# ── Database management ──────────────────────────────────────────────
@app.get("/api/faces")
async def list_faces():
    return {"faces": db.get_all_faces(), "count": db.count}


@app.delete("/api/faces/{face_id}")
async def delete_face(face_id: int):
    existing = db.get_all_faces()
    if not any(f["id"] == face_id for f in existing):
        raise HTTPException(404, f"Face ID {face_id} not found")
    db.delete_face(face_id)
    return {"message": f"Face {face_id} deleted"}


# ── Health / info ────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    import os
    return {
        "status": "ok" if processor.ready else "error",
        "models_ready": _models_ready,
        "model_loaded": processor.ready,
        "model_error": _model_error,
        "face_model": os.environ.get("FACE_MODEL", "default"),
        "db_faces": db.count,
    }


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

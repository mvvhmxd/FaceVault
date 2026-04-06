import io
import base64
import json
import cv2
import numpy as np
from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Request

router = APIRouter()


def _decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image data")
    return img


def _decode_b64(data: str) -> np.ndarray:
    if ',' in data:
        data = data.split(',', 1)[1]
    return _decode_upload(base64.b64decode(data))


def _encode_b64(image: np.ndarray, fmt: str = '.jpg') -> str:
    _, buf = cv2.imencode(fmt, image)
    return base64.b64encode(buf).decode()


async def _get_image(file: Optional[UploadFile], image_data: Optional[str]) -> np.ndarray:
    if file is not None:
        return _decode_upload(await file.read())
    if image_data:
        return _decode_b64(image_data)
    raise HTTPException(400, "No image provided")


# ── Full pipeline ────────────────────────────────────────────────────

@router.post("/process-frame")
async def process_frame(
    request: Request,
    file: UploadFile = File(None),
    image_data: str = Form(None),
):
    s = request.app.state
    img = await _get_image(file, image_data)
    faces = s.face_engine.detect_faces(img)

    results = []
    for face in faces:
        identity = None
        if face.embedding:
            identity = s.database.find_match(face.embedding, threshold=0.4)

        x1, y1, x2, y2 = face.bbox
        h, w = img.shape[:2]
        crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        emo = s.emotion.predict(crop)
        live = s.liveness.check(img, face.bbox)

        celeb = []
        if face.embedding and s.celebrity.count > 0:
            celeb = s.celebrity.find_match(face.embedding, top_k=1)

        results.append({
            'bbox': face.bbox,
            'name': identity['name'] if identity else 'Unknown',
            'recognition_confidence': round(identity['score'], 3) if identity else 0,
            'detection_confidence': round(face.confidence, 3),
            'age': face.age,
            'gender': face.gender,
            'emotion': emo.get('emotion', 'neutral'),
            'emotion_confidence': round(emo.get('confidence', 0), 3),
            'is_live': live.get('is_live', True),
            'liveness_confidence': round(live.get('confidence', 0.5), 3),
            'liveness_checks': live.get('checks', {}),
            'celebrity_match': celeb[0] if celeb else None,
        })

    return {'faces': results, 'count': len(results)}


# ── Registration ─────────────────────────────────────────────────────

@router.post("/register")
async def register_face(
    request: Request,
    name: str = Form(...),
    file: UploadFile = File(None),
    image_data: str = Form(None),
):
    s = request.app.state
    img = await _get_image(file, image_data)
    faces = s.face_engine.detect_faces(img)

    if not faces:
        raise HTTPException(400, "No face detected in image")
    face = faces[0]
    if face.embedding is None:
        raise HTTPException(400, "Could not extract embedding")

    fid = s.database.add_face(
        name=name,
        embedding=face.embedding,
        metadata={'age': face.age, 'gender': face.gender},
    )

    x1, y1, x2, y2 = face.bbox
    h, w = img.shape[:2]
    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    s.reconstruction.add_reference(fid, np.array(face.embedding), crop)

    return {'success': True, 'id': fid, 'name': name}


# ── Embedding extraction ────────────────────────────────────────────

@router.post("/extract-embedding")
async def extract_embedding(
    request: Request,
    file: UploadFile = File(None),
    image_data: str = Form(None),
):
    img = await _get_image(file, image_data)
    faces = request.app.state.face_engine.detect_faces(img)
    if not faces:
        raise HTTPException(400, "No face detected")
    return {
        'embeddings': [
            {'bbox': f.bbox, 'embedding': f.embedding, 'dim': len(f.embedding)}
            for f in faces if f.embedding
        ],
    }


# ── Embedding comparison ────────────────────────────────────────────

@router.post("/compare-embeddings")
async def compare_embeddings(data: str = Form(...)):
    from engine.similarity import EmbeddingSimilarity
    parsed = json.loads(data)
    e1, e2 = parsed.get('embedding1'), parsed.get('embedding2')
    if not e1 or not e2:
        raise HTTPException(400, "Two embeddings required")
    sim = EmbeddingSimilarity.cosine_similarity(e1, e2)
    dist = EmbeddingSimilarity.euclidean_distance(e1, e2)
    return {
        'cosine_similarity': round(sim, 4),
        'euclidean_distance': round(dist, 4),
        'is_same_person': sim > 0.4,
        'confidence': round(min(sim * 1.5, 1.0), 4),
    }


# ── Reconstruction ──────────────────────────────────────────────────

@router.post("/reconstruct-face")
async def reconstruct_face(
    request: Request,
    file: UploadFile = File(None),
    image_data: str = Form(None),
    embedding_data: str = Form(None),
):
    s = request.app.state
    embedding, original_crop = None, None

    if embedding_data:
        embedding = np.array(json.loads(embedding_data))
    else:
        img = await _get_image(file, image_data)
        faces = s.face_engine.detect_faces(img)
        if not faces or not faces[0].embedding:
            raise HTTPException(400, "No face detected")
        embedding = np.array(faces[0].embedding)
        x1, y1, x2, y2 = faces[0].bbox
        h, w = img.shape[:2]
        original_crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    result = s.reconstruction.reconstruct(embedding)
    if not result['success']:
        return result

    quality = 0.0
    if original_crop is not None:
        quality = s.reconstruction.compute_quality(original_crop, result['image'])

    return {
        'success': True,
        'reconstructed_image': f"data:image/jpeg;base64,{_encode_b64(result['image'])}",
        'similarity': round(result['similarity'], 4),
        'quality': round(quality, 4),
        'method': result['method'],
        'contributors': result['contributors'],
    }


# ── Privacy score ────────────────────────────────────────────────────

@router.post("/privacy-score")
async def privacy_score(
    request: Request,
    file: UploadFile = File(None),
    image_data: str = Form(None),
    embedding_data: str = Form(None),
):
    s = request.app.state

    if embedding_data:
        emb = np.array(json.loads(embedding_data))
    else:
        img = await _get_image(file, image_data)
        faces = s.face_engine.detect_faces(img)
        if not faces or not faces[0].embedding:
            raise HTTPException(400, "No face detected")
        emb = np.array(faces[0].embedding)

    recon = s.reconstruction.reconstruct(emb)
    rq = recon.get('similarity') if recon['success'] else None
    return s.privacy.compute_privacy_score(emb, reconstruction_quality=rq)


# ── Spoof check ──────────────────────────────────────────────────────

@router.post("/spoof-check")
async def spoof_check(
    request: Request,
    file: UploadFile = File(None),
    image_data: str = Form(None),
):
    s = request.app.state
    img = await _get_image(file, image_data)
    faces = s.face_engine.detect_faces(img)
    if not faces:
        raise HTTPException(400, "No face detected")
    result = s.liveness.check(img, faces[0].bbox)
    return {
        'is_genuine': result['is_live'],
        'confidence': result['confidence'],
        'checks': result['checks'],
        'details': result['details'],
    }


# ── Database CRUD ────────────────────────────────────────────────────

@router.get("/faces")
async def list_faces(request: Request):
    faces = request.app.state.database.get_all_faces()
    return {'faces': faces, 'count': len(faces)}


@router.delete("/faces/{face_id}")
async def delete_face(face_id: int, request: Request):
    if request.app.state.database.delete_face(face_id):
        return {'success': True}
    raise HTTPException(404, "Face not found")


# ── System stats ─────────────────────────────────────────────────────

@router.get("/stats")
async def stats(request: Request):
    s = request.app.state
    return {
        'registered_faces': len(s.database.get_all_faces()),
        'celebrity_count': s.celebrity.count,
        'reconstruction_refs': s.reconstruction.reference_count,
        'engines': {
            'face_detection': s.face_engine.ready,
            'emotion': True,
            'liveness': s.liveness.ready,
            'celebrity': s.celebrity.count > 0,
            'reconstruction': True,
            'privacy': True,
        },
    }

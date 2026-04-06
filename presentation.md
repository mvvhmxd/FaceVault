# FaceVault — Real-Time Face Recognition System
### Hackathon 2026 Presentation

---

## The Problem

- Face recognition is everywhere, but most demos are fragile scripts
- No explainability: users don't understand *what* the model sees
- No privacy awareness: embeddings leak identity information
- No anti-spoofing: trivially fooled by photos or screens

---

## Our Solution: FaceVault

A **production-grade, real-time face recognition system** with:
- Instant face detection and identification
- Live registration — no restart, no CLI
- Embedding explainability and reconstruction
- Quantified privacy risk analysis
- Multi-signal liveness / anti-spoofing

---

## Architecture

```
Browser (Webcam + Canvas Overlay)
        │
        ▼ base64 frames over HTTP
FastAPI Backend
  ├── InsightFace ArcFace (detect, embed, age, gender)
  ├── MediaPipe FaceMesh (emotion, blink, liveness)
  ├── Celebrity Matcher (embedding similarity)
  ├── Reconstruction Engine (nearest-blend + PCA)
  ├── Privacy Analyzer (leakage scoring)
  └── SQLite (embeddings + thumbnails)
```

---

## Key Features

1. **Real-time recognition** — 5–15 FPS, CPU only
2. **Live face registration** — from the browser, instant
3. **Emotion detection** — 7 emotions from facial landmarks
4. **Age & gender** — per-face estimation
5. **Liveness detection** — blink + motion + texture + frequency + color
6. **Celebrity look-alike** — embedding-based matching
7. **Multi-face handling** — simultaneous detection
8. **Confidence scoring** — per-face recognition confidence

---

## Innovations

### Embedding Reconstruction
- Reconstruct a face from its 512-dim embedding alone
- Uses nearest-neighbor blending or PCA decomposition
- Side-by-side comparison with original

### Privacy Risk Scoring
- Quantifies how much identity info leaks from an embedding
- Risk levels: minimal → low → moderate → high → critical
- Human-readable explanations + actionable recommendations

### Anti-Spoofing Pipeline
- 5 independent checks: texture, color, blink, motion, frequency
- Weighted combination for robust spoof detection
- Detects static image and replay attacks

### Identity Interpolation
- Smoothly blend between two face embeddings
- Visual output shows transition through latent space

---

## Demo Flow

1. **Open app** → dark, polished UI loads
2. **Start camera** → real-time face detection begins
3. **See overlays** → bounding boxes, name, emotion, age
4. **Register a face** → click button, enter name, saved instantly
5. **Recognition** → face now identified with confidence score
6. **Reconstruct** → see what the model "remembers" from the embedding
7. **Privacy check** → quantified risk analysis with recommendations
8. **Celebrity match** → find your look-alike

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python, FastAPI, uvicorn |
| Face AI | InsightFace (ArcFace), ONNX Runtime |
| Landmarks | MediaPipe Face Mesh |
| Database | SQLite |
| Frontend | Vanilla JS, Canvas API |
| Deploy | Vercel-compatible |

---

## Challenges

- Balancing model accuracy vs. CPU performance
- Real-time emotion detection without heavy frameworks (solved: landmark heuristics)
- Meaningful reconstruction without generative models (solved: nearest-blend)
- Privacy scoring that's both accurate and explainable

---

## Future Work

- WebRTC / WebSocket for lower-latency streaming
- On-device inference with WASM/WebGPU
- StyleGAN2 integration for photorealistic reconstruction
- Federated learning for privacy-preserving registration
- FAISS integration for large-scale identity databases
- Differential privacy for embedding storage

---

## Thank You

**FaceVault** — Face recognition that's fast, explainable, and privacy-aware.

Repository: `hackathon2026/`
Run: `python main.py` → `http://localhost:8000`

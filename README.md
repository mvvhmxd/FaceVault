# FaceVault — Real-Time Face Recognition System

A polished, real-time face recognition web application with advanced embedding intelligence, explainability, and biometric security — built to impress.

---

## Features

| Feature | Description |
|---|---|
| **Real-Time Recognition** | Webcam-based face detection and identification at ~5-15 FPS on CPU |
| **Live Registration** | Register new faces directly from the browser — no restart needed |
| **Emotion Detection** | Landmark-based emotion analysis (happy, sad, surprise, angry, neutral, fear, disgust) |
| **Age & Gender** | Predicted from ArcFace model per-face |
| **Liveness Detection** | Multi-signal anti-spoofing: blink, motion, texture, frequency, color analysis |
| **Celebrity Match** | Embedding-based look-alike matching with 20+ celebrity identities |
| **Embedding Reconstruction** | Nearest-neighbor and PCA-based face reconstruction from embeddings |
| **Privacy Risk Analysis** | Quantified privacy leakage scoring with human-readable explanations |
| **Spoof Detection** | Combined liveness + embedding consistency verification |
| **Multi-Face** | Simultaneous detection and recognition of multiple faces |
| **Identity Interpolation** | Blend between two face embeddings with visual output |

## Architecture

```
┌──────────────────────────────────────────────┐
│                BROWSER                       │
│  Webcam → Canvas Overlay → Face Cards        │
│  Registration Modal │ Reconstruction View    │
└────────────────┬─────────────────────────────┘
                 │ HTTP/JSON (base64 frames)
┌────────────────▼─────────────────────────────┐
│            FastAPI Backend (FaceVault)        │
│                                              │
│  /api/process-frame    (main pipeline)       │
│  /api/register         (add identity)        │
│  /api/extract-embedding                      │
│  /api/compare-embeddings                     │
│  /api/reconstruct-face                       │
│  /api/privacy-score                          │
│  /api/spoof-check                            │
│  /api/celebrity-match                        │
│  /api/interpolate                            │
│  /api/faces            (CRUD)                │
│                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │InsightFace│ │MediaPipe │ │Celebrity │     │
│  │ArcFace   │ │FaceMesh  │ │Matcher   │     │
│  │(detect,  │ │(emotion, │ │          │     │
│  │ embed,   │ │ liveness,│ │          │     │
│  │ age,     │ │ blink)   │ │          │     │
│  │ gender)  │ │          │ │          │     │
│  └──────────┘ └──────────┘ └──────────┘     │
│  ┌──────────────────────────────────────┐    │
│  │  Reconstruction │ Privacy │ Spoof    │    │
│  └──────────────────────────────────────┘    │
│                                              │
│  SQLite Database (embeddings + thumbnails)    │
└──────────────────────────────────────────────┘
```

## Tech Stack

- **Backend:** Python 3.10+, FastAPI, uvicorn
- **Face AI:** InsightFace (ArcFace / buffalo_l), MediaPipe Face Mesh
- **Inference:** ONNX Runtime (CPU)
- **Database:** SQLite (embeddings + metadata)
- **Frontend:** Vanilla HTML/CSS/JS, Canvas API
- **Deployment:** Vercel-compatible

## Quick Start

### 1. Clone and install

```bash
git clone <repo>
cd hackathon2026
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Run

```bash
python main.py
```

Open **http://localhost:8000** in your browser.

### 3. Use

1. Click **Start Camera** — allow webcam access
2. Faces are detected, analyzed, and displayed in real time
3. Click **Register Face** to save a new identity
4. Recognized faces show name + confidence; unknown faces show "Unknown"
5. Click **Reconstruct** to see embedding-based face reconstruction
6. Click **Privacy** to run privacy risk analysis

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/process-frame` | POST | Full pipeline: detect → recognize → emotion → liveness |
| `/api/register` | POST | Register new face (name + image) |
| `/api/extract-embedding` | POST | Extract 512-dim ArcFace embedding |
| `/api/compare-embeddings` | POST | Compare two embeddings (cosine similarity) |
| `/api/reconstruct-face` | POST | Reconstruct face from embedding |
| `/api/privacy-score` | POST | Privacy leakage analysis |
| `/api/spoof-check` | POST | Anti-spoofing verification |
| `/api/celebrity-match` | POST | Celebrity look-alike matching |
| `/api/interpolate` | POST | Interpolate between two face embeddings |
| `/api/faces` | GET | List registered identities |
| `/api/faces/{id}` | DELETE | Remove registered identity |
| `/api/health` | GET | Health check |

## Deployment

### Vercel

```bash
vercel --prod
```

### Docker (optional)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Performance

- Frames resized to 640px before processing
- Client-side frame skipping (200ms interval)
- Embedding cache in RAM for instant lookups
- Lightweight models only (CPU, no GPU required)
- Typical latency: 100–300ms per frame on modern CPU

## Troubleshooting

| Issue | Fix |
|---|---|
| Camera not starting | Check browser permissions; use HTTPS for remote access |
| "No face detected" | Ensure face is well-lit and centered |
| Slow performance | Reduce camera resolution; increase PROCESS_INTERVAL in app.js |
| Model download fails | Check internet; InsightFace downloads models on first run |
| Import errors | Ensure all dependencies installed: `pip install -r requirements.txt` |

## License

MIT

# FaceVault -- Technical Deep Dive & Study Guide

A complete map of every computer vision concept used in FaceVault, where it lives in the codebase, what models power it, and how to explain it to judges.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Face Detection](#2-face-detection)
3. [Face Embeddings & Recognition](#3-face-embeddings--recognition)
4. [Age & Gender Estimation](#4-age--gender-estimation)
5. [Emotion Detection](#5-emotion-detection)
6. [Liveness & Anti-Spoofing](#6-liveness--anti-spoofing)
7. [Celebrity Look-Alike Matching](#7-celebrity-look-alike-matching)
8. [Embedding Reconstruction](#8-embedding-reconstruction)
9. [Privacy Risk Analysis](#9-privacy-risk-analysis)
10. [Database & Similarity Search](#10-database--similarity-search)
11. [Frontend Pipeline](#11-frontend-pipeline)
12. [Key Formulas to Know](#12-key-formulas-to-know)
13. [Judge Q&A Cheat Sheet](#13-judge-qa-cheat-sheet)

---

## 1. System Architecture

```
Webcam (Browser)
    |
    | base64 JPEG frame (every 200ms)
    v
FastAPI Server (main.py)
    |
    +---> FaceProcessor (InsightFace / ONNX Runtime)
    |       |---> Face Detection (RetinaFace)
    |       |---> Embedding Extraction (ArcFace ResNet-50)
    |       |---> Age & Gender (CNN)
    |
    +---> EmotionDetector (MediaPipe FaceLandmarker)
    |       |---> 478 Face Landmarks
    |       |---> 52 Blendshape Coefficients
    |       |---> Rule-based Emotion Scoring
    |
    +---> LivenessDetector (OpenCV + MediaPipe)
    |       |---> Texture Analysis (LBP + Laplacian)
    |       |---> Moire Pattern Detection (FFT)
    |       |---> Screen Reflection Analysis (HSV)
    |       |---> Blink Detection (Blendshapes)
    |       |---> Motion Consistency (Frame Diff)
    |       |---> Frequency Analysis (FFT)
    |       |---> Context Analysis (Canny Edges)
    |       |---> Face Size Heuristic
    |
    +---> CelebrityMatcher (Cosine Similarity)
    +---> ReconstructionEngine (Nearest-Neighbor Blend / PCA)
    +---> PrivacyAnalyzer (Leakage Scoring)
    |
    v
SQLite Database (embeddings + thumbnails)
```

**Key insight for judges:** Every model runs on CPU via ONNX Runtime. No GPU required. Total model size ~330MB.

---

## 2. Face Detection

**Concept:** Locate all faces in an image and return bounding boxes.

**Model:** RetinaFace (via InsightFace `det_10g.onnx`, 16MB)

**How it works:**
- Single-stage anchor-based detector (like SSD/YOLO family)
- Feature Pyramid Network (FPN) for multi-scale detection
- Simultaneously predicts: bounding box, confidence score, 5 facial keypoints
- Input is resized to 640x640 before detection

**Code location:** `engine/face_processor.py`

```python
# Line 65: Detection happens here
faces = app.get(resized)

# Each face object contains:
face.bbox       # [x1, y1, x2, y2] bounding box
face.det_score  # confidence 0-1
face.kps        # 5 keypoints: left eye, right eye, nose, left mouth, right mouth
```

**CV concepts to know:**
- **Bounding box regression** -- predicting (x, y, w, h) offsets from anchor boxes
- **Non-Maximum Suppression (NMS)** -- removing duplicate detections
- **Feature Pyramid Network** -- detecting faces at multiple scales
- **Anchor boxes** -- predefined box shapes the detector refines
- **IoU (Intersection over Union)** -- measures overlap between predicted and ground truth boxes

---

## 3. Face Embeddings & Recognition

**Concept:** Convert a face image into a fixed-size vector (embedding) that captures identity.

**Model:** ArcFace with ResNet-50 backbone (`w600k_r50.onnx`, 166MB)

**How it works:**
1. Face is aligned using the 5 keypoints (affine transformation to standard pose)
2. Aligned 112x112 face is passed through ResNet-50
3. Output: 512-dimensional L2-normalized embedding vector
4. Two faces of the same person have HIGH cosine similarity (>0.4)
5. Two faces of different people have LOW similarity (<0.4)

**Code location:** `engine/face_processor.py`

```python
# Line 79: Embedding is extracted per-face
"embedding": face.embedding.astype(np.float32)  # shape: (512,)
```

**Recognition flow** (`engine/database.py`, line 67):
```python
def find_match(self, embedding, threshold=0.4):
    # Compare against every stored embedding using cosine similarity
    score = np.dot(embedding, stored_emb) / (norm_a * norm_b)
    # If score >= 0.4, it's a match
```

**CV concepts to know:**
- **Face alignment** -- affine transform using eye/nose positions to normalize pose
- **Metric learning** -- training embeddings so same-person pairs are close, different-person pairs are far
- **ArcFace loss** -- Additive Angular Margin Loss; adds a margin in angular space for better class separation
- **Cosine similarity** -- measures angle between two vectors: `cos(theta) = (A . B) / (|A| * |B|)`
- **Embedding space** -- high-dimensional space where geometric distance = identity similarity
- **L2 normalization** -- project embeddings onto unit hypersphere so cosine similarity = dot product
- **Threshold** -- 0.4 is standard for ArcFace; above = same person, below = different

---

## 4. Age & Gender Estimation

**Concept:** Predict age (integer) and gender (M/F) from a face.

**Model:** InsightFace `genderage.onnx` (1.3MB) -- lightweight CNN

**Code location:** `engine/face_processor.py`

```python
# Lines 80-81: Directly from InsightFace
"age": int(face.age),          # integer, e.g., 28
"gender": "Male" if face.sex == "M" else "Female"
```

**CV concepts to know:**
- **Multi-task learning** -- single network predicts both age and gender
- **Regression vs Classification** -- age is regression (continuous), gender is binary classification
- **Apparent age** -- model predicts perceived age, not actual age (can differ by +/-10 years)

---

## 5. Emotion Detection

**Concept:** Classify facial expression into one of 7 emotions.

**Model:** MediaPipe FaceLandmarker (`face_landmarker.task`, 4MB)

**How it works:**
1. Detect 478 face landmarks (3D mesh)
2. Extract 52 blendshape coefficients (FACS-like Action Units)
3. Map blendshapes to emotions using weighted rules:

```
happy    = mouthSmile * 0.6 + eyeSquint * 0.2
surprise = jawOpen * 0.4 + eyeWide * 0.4 + browUp * 0.3
sad      = mouthFrown * 0.5 + browUp * 0.15
angry    = browDown * 0.5 + noseSneer * 0.3
fear     = eyeWide * 0.4 + browUp * 0.2 + jawOpen * 0.15
disgust  = noseSneer * 0.5 + mouthFrown * 0.2
neutral  = 1.0 - max(others) * 1.5
```

4. Normalize scores to sum to 1.0 (softmax-like)

**Code location:** `engine/emotion.py`

```python
# Line 74: Primary method using blendshapes
def _detect_with_blendshapes(self, face_img, landmarker)

# Line 128: Fallback using pixel variance
def _detect_with_geometry(self, face_img, landmarker)
```

**CV concepts to know:**
- **Facial landmarks** -- 478 3D points describing face geometry
- **Blendshapes / FACS** -- Facial Action Coding System; decomposes expressions into Action Units (AU)
- **Action Units** -- atomic facial movements (AU6 = cheek raise, AU12 = lip corner pull = smile)
- **Rule-based classifier** -- maps continuous blendshape values to discrete emotion labels via thresholds

---

## 6. Liveness & Anti-Spoofing

**Concept:** Determine if the face is a real person or a photo/screen attack.

**Approach:** 9 independent checks, weighted and combined.

**Code location:** `engine/liveness.py`

### Check 1: Texture Analysis (LBP)
```
File: liveness.py, line 156
Concept: Local Binary Patterns
How: Compare each pixel to 8 neighbors, encode as 8-bit binary number
Why: Real skin has rich, varied texture. Screens/paper are smoother.
Metric: Variance of LBP values + Laplacian variance
```

### Check 2: Moire Pattern Detection
```
File: liveness.py, line 193
Concept: 2D Fast Fourier Transform (FFT)
How: Convert face to frequency domain, analyze high-frequency energy
Why: Phone/monitor screens have periodic pixel grids that create moire patterns
      visible as peaks in the frequency spectrum
Metric: Peak ratio in high-frequency band, horizontal/vertical line strength
```

### Check 3: Screen Reflection Analysis
```
File: liveness.py, line 240
Concept: HSV color space analysis
How: Detect bright spots (V>220) with low saturation (screen glare)
     Measure brightness uniformity across blocks (screens have even backlight)
     Count unique color values (screens have limited bit depth)
Why: Screens produce uniform specular highlights that skin doesn't
```

### Check 4: Color Distribution
```
File: liveness.py, line 286
Concept: Skin color modeling in HSV space
How: Check skin-tone ratio, saturation variance, blue channel dominance
Why: Screens emit more blue light; real skin has natural color variation
```

### Check 5: Blink Detection (Temporal)
```
File: liveness.py, line 318
Concept: Blendshape-based Eye Aspect Ratio
How: Track eyeBlinkLeft/eyeBlinkRight blendshapes over time
     Maintain per-face blink history (up to 60 frames)
     A photo NEVER blinks
Why: Strongest single anti-spoofing signal. Once blink detected = proven live.
State: Per-face tracking using _FaceTrack objects, keyed by quantized bbox position
```

### Check 6: Motion Consistency (Temporal)
```
File: liveness.py, line 367
Concept: Frame differencing
How: Compute mean absolute pixel difference between consecutive face crops
     Track variance of motion over 30 frames
Why: Real faces have natural micro-movements; photos are perfectly still
```

### Check 7: Frequency Analysis
```
File: liveness.py, line 396
Concept: FFT spectral analysis
How: Compute center-to-outer energy ratio and periodic spike detection
Why: Natural faces have smooth frequency falloff; screens have periodic artifacts
```

### Check 8: Context Analysis
```
File: liveness.py, line 453
Concept: Edge detection + color temperature
How: Expand bbox by 60%, run Canny edge detector on border region
     Check brightness contrast between face and surroundings
     Compare blue/red ratio (color temperature) of face vs background
Why: Phone screens have visible bezels, different lighting, and different color temp
```

### Check 9: Face Size Check
```
File: liveness.py, line 430
Concept: Geometric heuristic
How: face_area / frame_area ratio
     Real face at webcam distance: 5-40% of frame
     Phone screen face: 1-5% of frame
```

### Scoring System

```python
Weights: texture=0.06, moire=0.10, reflection=0.06, color=0.05,
         blink=0.25, motion=0.13, frequency=0.10, context=0.15, size=0.10

Boosts:
  - ever_blinked = True  --> score >= 0.62 (guaranteed live)
  - blink > 0.85         --> score >= 0.70

Vetoes (hard caps):
  - 15 frames, no blink  --> score <= 0.45
  - 30 frames, no blink  --> score <= 0.35
  - context < 0.2        --> score <= 0.30
  - moire + reflection both < 0.2 --> score <= 0.30

Threshold: score > 0.50 = LIVE
```

**CV concepts to know:**
- **Local Binary Patterns (LBP)** -- texture descriptor, rotation/illumination invariant
- **Fourier Transform / FFT** -- decomposes image into frequency components
- **Moire patterns** -- interference patterns from overlapping periodic structures
- **HSV color space** -- separates Hue, Saturation, Value; better for color analysis than RGB
- **Canny edge detection** -- multi-stage edge detector (Gaussian blur, gradient, NMS, hysteresis)
- **Temporal analysis** -- using multiple frames over time to detect blinks and motion
- **Eye Aspect Ratio (EAR)** -- ratio of eye height to width; drops during blinks

---

## 7. Celebrity Look-Alike Matching

**Concept:** Find the most similar celebrity from a pre-computed database.

**How it works:**
1. 40 celebrities from LFW dataset, embeddings pre-computed with ArcFace
2. Each celebrity has: average embedding (512-dim) + face thumbnail (80x80 JPEG)
3. For a query face, compute cosine similarity against all 40 celebrities
4. Return top-K sorted by similarity

**Code location:** `engine/celebrity.py`

```python
# Line 39: Core matching
sim = np.dot(embedding, ce) / (en * np.linalg.norm(ce) + 1e-6)
```

**Building the database:** `build_celeb_db.py`
- Load images, upscale to 400px (LFW images are tiny)
- Run InsightFace detection + embedding
- Average embeddings across multiple photos per person
- L2-normalize the averaged embedding
- Store as JSON with base64 thumbnails

**CV concepts to know:**
- **K-Nearest Neighbors (KNN)** -- finding closest vectors in embedding space
- **Cosine similarity vs Euclidean distance** -- cosine ignores magnitude, measures angle
- **Embedding averaging** -- multiple photos per person averaged and re-normalized for robustness

---

## 8. Embedding Reconstruction

**Concept:** Reconstruct what a face "looks like" from its 512-dim embedding alone.

**Code location:** `engine/reconstruction.py`

### Method 1: Nearest-Neighbor Blend (line 18)
```
1. Find K=5 closest faces in database by cosine similarity
2. Load their stored thumbnails (96x96)
3. Blend pixels weighted by similarity^2 (squared for emphasis on closest matches)
4. Result: blurred composite showing model's "understanding" of the face
```

### Method 2: PCA Reconstruction (line 56)
```
1. Stack all stored embeddings into matrix
2. Compute SVD (Singular Value Decomposition): U, S, Vt = SVD(centered)
3. Project query embedding onto top-10 principal components
4. Reconstruct: mean + projection * components
5. Measure reconstruction quality via cosine similarity
```

### Method 3: Embedding Visualization (line 84)
```
When no faces in database:
1. Normalize embedding to [0, 1]
2. Reshape into sqrt(512) x sqrt(512) grid
3. Upscale to 96x96
4. Apply VIRIDIS colormap
```

**CV concepts to know:**
- **PCA (Principal Component Analysis)** -- dimensionality reduction, finds directions of maximum variance
- **SVD (Singular Value Decomposition)** -- matrix factorization: M = U * S * Vt
- **Image blending** -- weighted pixel averaging of multiple images
- **Latent space** -- compressed representation space where similar faces are nearby
- **Reconstruction attack** -- recovering original data from its compressed representation

---

## 9. Privacy Risk Analysis

**Concept:** Quantify how much personal information leaks from a stored embedding.

**Code location:** `engine/privacy.py`

### Metrics Computed:

**Leakage Score** (line 10):
```
How similar is the reconstruction to the original?
leakage = reconstruction_similarity  (0 to 1)
Higher = more privacy risk
```

**Embedding Entropy** (line 25):
```
Shannon entropy of embedding values treated as probability distribution:
H = -sum(p * log2(p)) / log2(N)
Normalized to [0, 1]
Higher entropy = more information stored = more risk
```

**Uniqueness Score** (line 31):
```
How different is this face from everyone else in the database?
uniqueness = 1 - mean(similarity to all stored faces)
Higher = more unique = potentially more identifiable
```

**Risk Levels:**
```
> 0.8 = CRITICAL
> 0.6 = HIGH
> 0.4 = MODERATE
> 0.2 = LOW
else  = MINIMAL
```

**CV concepts to know:**
- **Information entropy** -- measures uncertainty/information content of a distribution
- **Privacy-preserving ML** -- techniques to protect biometric data
- **Differential privacy** -- adding noise to prevent individual re-identification
- **Biometric template protection** -- securing stored embeddings

---

## 10. Database & Similarity Search

**Code location:** `engine/database.py`

**Storage:** SQLite + in-memory cache (Python dict)

```
Schema:
  id        INTEGER PRIMARY KEY
  name      TEXT
  embedding BLOB (512 * 4 = 2048 bytes as float32)
  thumbnail BLOB (JPEG compressed face crop)
  metadata  TEXT (JSON: age, gender)
  created_at TIMESTAMP
```

**Search:** Brute-force cosine similarity (fine for <1000 faces)

**Duplicate detection** (line 56):
- `find_duplicate_face(threshold=0.75)` -- prevents registering the same face twice
- `name_exists()` -- prevents duplicate names (case-insensitive)

**CV concepts to know:**
- **FAISS** -- Facebook's library for billion-scale similarity search (mentioned as upgrade path)
- **Cosine similarity search** -- O(N) brute force, O(log N) with approximate methods
- **Embedding quantization** -- compressing float32 to int8 for storage efficiency

---

## 11. Frontend Pipeline

**Code location:** `static/js/app.js`

```
Loop (every 200ms):
  1. Capture video frame from webcam (getUserMedia API)
  2. Draw to hidden canvas
  3. Export as JPEG base64 (0.7 quality)
  4. POST to /api/process-frame as FormData
  5. Receive JSON with face detections
  6. Draw bounding boxes + labels on overlay canvas
  7. Update face info cards in side panel
  8. Update FPS counter
```

**Canvas overlay drawing** (line 113):
- Rounded bounding boxes with corner accents
- Name label with colored background (green=known, amber=unknown)
- Emotion + age tag at bottom of box
- Liveness indicator dot (green=live, red=spoof)

---

## 12. Key Formulas to Know

### Cosine Similarity
```
sim(A, B) = (A . B) / (|A| * |B|)

Where A . B = sum(Ai * Bi) for i in 1..512
|A| = sqrt(sum(Ai^2))

Range: [-1, 1], threshold for same person: 0.4
```

### ArcFace Loss
```
L = -log( exp(s * cos(theta_yi + m)) / (exp(s * cos(theta_yi + m)) + sum_j exp(s * cos(theta_j))) )

s = scale factor (64)
m = angular margin (0.5 radians)
theta_yi = angle between embedding and true class center
```

### Eye Aspect Ratio (EAR)
```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

p1-p6 = eye landmark points
EAR < 0.21 = blink detected
```

### Local Binary Pattern (LBP)
```
For each pixel, compare with 8 neighbors:
LBP = sum( (neighbor >= center) * 2^i ) for i in 0..7

Result: 8-bit code (0-255) per pixel
Variance of all LBP codes = texture measure
```

### Shannon Entropy
```
H = -sum( p(x) * log2(p(x)) ) for all x

Normalized: H_norm = H / log2(N)
Range: [0, 1]
```

### Privacy Leakage
```
leakage = cosine_similarity(original_embedding, reconstructed_embedding)
risk = "critical" if leakage > 0.8
```

---

## 13. Judge Q&A Cheat Sheet

**Q: How does face recognition work?**
A: We extract a 512-dimensional embedding using ArcFace (ResNet-50 backbone). This maps each face to a point in embedding space where distance = identity similarity. We compare new faces against stored embeddings using cosine similarity with a 0.4 threshold.

**Q: Why ArcFace and not FaceNet or DeepFace?**
A: ArcFace uses Additive Angular Margin Loss which pushes inter-class boundaries harder than triplet loss (FaceNet) or softmax (DeepFace). It achieves 99.83% on LFW benchmark. We use InsightFace's implementation which bundles detection + alignment + recognition.

**Q: How does liveness detection work?**
A: Nine independent checks: texture analysis (LBP), moire detection (FFT), screen reflection (HSV), color analysis, blink detection (temporal blendshapes), motion consistency (frame differencing), frequency analysis, context analysis (edge detection around face), and face size heuristic. The key insight: a photo never blinks. After 15+ frames without a blink, the score is hard-capped.

**Q: Can you explain the reconstruction module?**
A: We reconstruct faces from embeddings using nearest-neighbor blending. Find the 5 closest faces in the database, blend their thumbnails weighted by similarity-squared. This shows what the model "remembers" from just 512 numbers. Higher reconstruction quality = higher privacy risk because more identity information is recoverable from the embedding alone.

**Q: How do you handle privacy?**
A: We store embeddings, not images (except small thumbnails for reconstruction demo). Our privacy analyzer computes a leakage score: how well can we reconstruct a face from its embedding? We report risk levels and provide actionable recommendations like differential privacy, embedding rotation, and encryption at rest.

**Q: Why not use a GPU?**
A: All models run on CPU via ONNX Runtime. InsightFace's buffalo_l is optimized for CPU inference. Total model size is 330MB, typical latency is 300-500ms per frame locally. This makes deployment simple (no CUDA dependency) and accessible on any machine.

**Q: How does celebrity matching work?**
A: We pre-computed ArcFace embeddings for 40 celebrities from the LFW dataset. For a query face, we compute cosine similarity against all 40 stored embeddings and return the top matches. Same embedding space means distances are directly comparable.

**Q: What happens with multiple faces?**
A: The detector finds all faces simultaneously. Each gets independent processing: embedding, emotion, liveness, celebrity match. Registration picks the largest face (by bounding box area). The UI shows info cards for every detected face.

**Q: How do you prevent duplicate registrations?**
A: Two checks: (1) name uniqueness (case-insensitive), (2) face embedding similarity > 0.75 against all stored faces. If the same face + same name, we update the embedding. If same face + different name, we reject with an explanation.

**Q: What's the accuracy?**
A: buffalo_l achieves 99.5% on LFW for face verification. Age estimation is +/-5 years. Emotion detection uses MediaPipe blendshapes with rule-based classification. Liveness detection catches photo attacks within 3-6 seconds via mandatory blink verification.

---

## File Map

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | ~350 | FastAPI server, 12 API endpoints, request validation |
| `engine/face_processor.py` | ~105 | InsightFace wrapper: detection, embedding, age, gender |
| `engine/emotion.py` | ~160 | MediaPipe blendshapes -> 7-class emotion |
| `engine/liveness.py` | ~545 | 9-check anti-spoofing with per-face temporal tracking |
| `engine/celebrity.py` | ~75 | Cosine similarity matching against 40 celebrity embeddings |
| `engine/reconstruction.py` | ~105 | Nearest-blend + PCA face reconstruction from embeddings |
| `engine/privacy.py` | ~85 | Entropy, uniqueness, leakage scoring |
| `engine/database.py` | ~145 | SQLite + RAM cache, duplicate detection, CRUD |
| `static/js/app.js` | ~440 | Webcam capture, canvas overlay, face cards, modals |
| `static/css/style.css` | ~340 | Dark theme, glass-morphism, responsive layout |
| `static/index.html` | ~160 | Page structure, modals, SVG icons |

---

## Models Used

| Model | File | Size | Framework | Purpose |
|-------|------|------|-----------|---------|
| RetinaFace | det_10g.onnx | 16 MB | ONNX Runtime | Face detection |
| ArcFace ResNet-50 | w600k_r50.onnx | 166 MB | ONNX Runtime | Face embeddings |
| 3D Landmarks | 1k3d68.onnx | 137 MB | ONNX Runtime | Face alignment |
| 2D Landmarks | 2d106det.onnx | 5 MB | ONNX Runtime | Face alignment |
| GenderAge | genderage.onnx | 1.3 MB | ONNX Runtime | Age + gender |
| FaceLandmarker | face_landmarker.task | 4 MB | MediaPipe | Emotion + liveness |

**Total: ~330 MB on disk, ~500 MB RAM during inference**

---

## How to Study This Project

1. **Start with `main.py`** -- read the `/api/process-frame` endpoint to see the full pipeline
2. **Read `face_processor.py`** -- understand how InsightFace wraps detection + embedding
3. **Read `database.py`** -- understand cosine similarity search and duplicate detection
4. **Read `emotion.py`** -- understand blendshapes and how they map to emotions
5. **Read `liveness.py`** -- the most complex module; study each check individually
6. **Read `reconstruction.py`** -- understand nearest-neighbor blending and PCA
7. **Read `privacy.py`** -- understand entropy and leakage scoring
8. **Read `app.js`** -- understand the frontend loop and canvas drawing
9. **Run locally** and experiment: register faces, show phone photos, try reconstruction
10. **Read the formulas** in section 12 until you can explain each one from memory

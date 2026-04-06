#!/usr/bin/env python3
"""Download lightweight datasets and pre-trained model artefacts for FaceVault.

Run once before starting the server:
    python scripts/setup_data.py

What it fetches
---------------
1. Emotion-FERPlus ONNX model (~34 MB)
2. LFW people sample (10 identities x 5 images via scikit-learn)
3. Generates synthetic celebrity embedding stubs so the demo works
   out of the box (replace with real CelebA embeddings for production).
"""

import os
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


def download_file(url: str, dest: Path, label: str = ""):
    if dest.exists():
        print(f"  [skip] {label or dest.name} already exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [download] {label or dest.name} ...")
    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"  [done] {size_mb:.1f} MB")


# ── 1. Emotion-FERPlus ONNX model ──────────────────────────────────

def fetch_emotion_model():
    print("\n=== Emotion Model (FERPlus ONNX) ===")
    url = (
        "https://github.com/onnx/models/raw/main/validated/"
        "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    )
    download_file(url, MODELS_DIR / "emotion-ferplus-8.onnx", "emotion-ferplus-8.onnx")


# ── 2. LFW Sample (via sklearn) ───────────────────────────────────

def fetch_lfw_sample():
    print("\n=== LFW Sample (via scikit-learn) ===")
    try:
        from sklearn.datasets import fetch_lfw_people
        import numpy as np

        lfw_dir = DATA_DIR / "lfw_sample"
        if lfw_dir.exists() and any(lfw_dir.iterdir()):
            print("  [skip] LFW sample already present")
            return

        print("  [download] Fetching LFW subset (min 5 images/person) ...")
        lfw = fetch_lfw_people(min_faces_per_person=5, resize=1.0)
        names = lfw.target_names
        lfw_dir.mkdir(parents=True, exist_ok=True)

        import cv2
        saved = 0
        for i, (img_arr, target) in enumerate(zip(lfw.images, lfw.target)):
            name = names[target].replace(" ", "_")
            person_dir = lfw_dir / name
            person_dir.mkdir(exist_ok=True)
            fname = person_dir / f"{i:04d}.jpg"
            if not fname.exists():
                # sklearn returns float 0-1; convert to uint8 0-255
                img_u8 = (img_arr * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(fname), img_u8)
                saved += 1
            if saved >= 50:
                break

        print(f"  [done] Saved {saved} LFW images")
    except ImportError:
        print("  [skip] scikit-learn not installed — skipping LFW download")
    except Exception as exc:
        print(f"  [warn] LFW download failed: {exc}")


# ── 3. Synthetic celebrity stub embeddings ─────────────────────────

CELEBRITY_NAMES = [
    "Tom Hanks", "Scarlett Johansson", "Leonardo DiCaprio",
    "Jennifer Lawrence", "Brad Pitt", "Angelina Jolie",
    "Robert Downey Jr", "Emma Watson", "Chris Hemsworth",
    "Natalie Portman", "Morgan Freeman", "Cate Blanchett",
    "Keanu Reeves", "Margot Robbie", "Denzel Washington",
    "Anne Hathaway", "Ryan Gosling", "Meryl Streep",
    "Will Smith", "Gal Gadot", "Johnny Depp",
    "Emma Stone", "Chris Evans", "Zendaya",
    "Timothee Chalamet", "Florence Pugh", "Oscar Isaac",
    "Pedro Pascal", "Jenna Ortega", "Sydney Sweeney",
]


def generate_celebrity_stubs():
    """Create random 512-d embedding stubs so celebrity matching
    works in demo mode. Replace with real embeddings extracted from
    CelebA or VGGFace2 images for production use."""
    print("\n=== Celebrity Embedding Stubs ===")
    out = DATA_DIR / "celebrities.json"
    if out.exists():
        print("  [skip] celebrities.json already exists")
        return

    import numpy as np
    np.random.seed(42)
    data = []
    for name in CELEBRITY_NAMES:
        emb = np.random.randn(512).astype(float)
        emb /= np.linalg.norm(emb)
        data.append({"name": name, "embedding": emb.tolist()})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f)
    print(f"  [done] Generated {len(data)} celebrity embedding stubs")


# ── 4. Ensure InsightFace model cache ──────────────────────────────

def warm_insightface():
    print("\n=== InsightFace Model Warm-up ===")
    try:
        from insightface.app import FaceAnalysis
        print("  [download] Downloading InsightFace buffalo_l (first run only) ...")
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("  [done] InsightFace ready")
    except Exception as exc:
        print(f"  [warn] InsightFace warm-up failed: {exc}")
        print("         Models will download on first server start.")


# ── main ───────────────────────────────────────────────────────────

def main():
    print("=" * 56)
    print("  FaceVault — Dataset & Model Setup")
    print("=" * 56)

    fetch_emotion_model()
    generate_celebrity_stubs()
    fetch_lfw_sample()
    warm_insightface()

    print("\n" + "=" * 56)
    print("  Setup complete!  Run the server:")
    print("    python main.py")
    print("=" * 56)


if __name__ == "__main__":
    main()

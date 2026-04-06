"""Build celebrity embedding database from LFW sample images."""
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

LFW_DIR = Path("data/lfw_sample")
OUT_FILE = Path("data/celebrities/celeb_db.json")


def main():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))

    db = {}
    for person_dir in sorted(LFW_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name.replace("_", " ")
        images = list(person_dir.glob("*.jpg"))
        if not images:
            continue

        embeddings = []
        best_thumb = None
        best_score = 0

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            scale = max(1, 400 / max(h, w))
            if scale > 1:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            faces = app.get(img)
            if not faces:
                continue

            f = faces[0]
            embeddings.append(f.embedding)

            if f.det_score > best_score:
                best_score = f.det_score
                bx = f.bbox.astype(int)
                bh, bw = img.shape[:2]
                x1, y1 = max(0, bx[0]), max(0, bx[1])
                x2, y2 = min(bw, bx[2]), min(bh, bx[3])
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    crop = cv2.resize(crop, (80, 80))
                    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    best_thumb = base64.b64encode(buf).decode()

        if not embeddings or best_thumb is None:
            print(f"  SKIP {name}")
            continue

        avg_emb = np.mean(embeddings, axis=0).astype(np.float32)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-6)

        db[name] = {
            "embedding": avg_emb.tolist(),
            "thumbnail": best_thumb,
        }
        print(f"  OK   {name} ({len(embeddings)} img)")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(db, f)
    print(f"\nSaved {len(db)} celebrities to {OUT_FILE}")


if __name__ == "__main__":
    main()

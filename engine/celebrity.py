import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CelebrityMatcher:
    def __init__(self, data_path="data/celebrities"):
        self.data_path = Path(data_path)
        self.celebrities: dict[str, dict] = {}
        self._load()

    def _load(self):
        db_file = self.data_path / "celeb_db.json"
        if db_file.exists():
            with open(db_file) as f:
                data = json.load(f)
            for name, entry in data.items():
                self.celebrities[name] = {
                    "embedding": np.array(entry["embedding"], dtype=np.float32),
                    "thumbnail": entry.get("thumbnail"),
                }
            logger.info(f"Loaded {len(self.celebrities)} celebrities with thumbnails")
            return

        legacy = self.data_path / "embeddings.json"
        if legacy.exists():
            with open(legacy) as f:
                data = json.load(f)
            for name, emb in data.items():
                self.celebrities[name] = {
                    "embedding": np.array(emb, dtype=np.float32),
                    "thumbnail": None,
                }
            logger.info(f"Loaded {len(self.celebrities)} celebrities (legacy, no thumbnails)")

    def match(self, embedding, top_k=3):
        if not self.celebrities:
            return []

        en = np.linalg.norm(embedding)
        scores = []
        for name, entry in self.celebrities.items():
            ce = entry["embedding"]
            sim = float(np.dot(embedding, ce) / (en * np.linalg.norm(ce) + 1e-6))
            scores.append({
                "name": name,
                "similarity": round(sim, 3),
                "thumbnail": entry.get("thumbnail"),
            })

        scores.sort(key=lambda x: -x["similarity"])
        return scores[:top_k]

    def add(self, name, embedding, thumbnail=None):
        self.celebrities[name] = {
            "embedding": embedding.astype(np.float32),
            "thumbnail": thumbnail,
        }
        self._save()

    def _save(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        data = {}
        for name, entry in self.celebrities.items():
            data[name] = {
                "embedding": entry["embedding"].tolist(),
                "thumbnail": entry.get("thumbnail"),
            }
        with open(self.data_path / "celeb_db.json", "w") as f:
            json.dump(data, f)

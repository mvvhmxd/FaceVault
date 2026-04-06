import sqlite3
import json
import numpy as np
from pathlib import Path
import threading
import time


class FaceDatabase:
    def __init__(self, db_path="data/faces.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()
        self._cache = {}
        self._load_cache()

    def _get_conn(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                thumbnail BLOB,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()

    def _load_cache(self):
        conn = self._get_conn()
        rows = conn.execute("SELECT id, name, embedding, thumbnail FROM faces").fetchall()
        for row in rows:
            emb = np.frombuffer(row[2], dtype=np.float32).copy()
            self._cache[row[0]] = {
                "name": row[1],
                "embedding": emb,
                "thumbnail": row[3],
            }

    def name_exists(self, name):
        return any(d["name"].lower() == name.lower() for d in self._cache.values())

    def find_duplicate_face(self, embedding, threshold=0.75):
        """Check if a very similar face is already registered."""
        if not self._cache:
            return None
        en = np.linalg.norm(embedding)
        for fid, data in self._cache.items():
            score = float(
                np.dot(embedding, data["embedding"])
                / (en * np.linalg.norm(data["embedding"]) + 1e-6)
            )
            if score >= threshold:
                return {"id": fid, "name": data["name"], "score": score}
        return None

    def add_face(self, name, embedding, thumbnail_bytes=None, metadata=None):
        conn = self._get_conn()
        emb_bytes = embedding.astype(np.float32).tobytes()
        meta_json = json.dumps(metadata) if metadata else None
        cursor = conn.execute(
            "INSERT INTO faces (name, embedding, thumbnail, metadata) VALUES (?, ?, ?, ?)",
            (name, emb_bytes, thumbnail_bytes, meta_json),
        )
        conn.commit()
        face_id = cursor.lastrowid
        self._cache[face_id] = {
            "name": name,
            "embedding": embedding.astype(np.float32),
            "thumbnail": thumbnail_bytes,
        }
        return face_id

    def update_face(self, face_id, embedding, thumbnail_bytes=None):
        """Update embedding (and optionally thumbnail) for an existing face."""
        conn = self._get_conn()
        emb_bytes = embedding.astype(np.float32).tobytes()
        if thumbnail_bytes is not None:
            conn.execute(
                "UPDATE faces SET embedding=?, thumbnail=? WHERE id=?",
                (emb_bytes, thumbnail_bytes, face_id),
            )
        else:
            conn.execute("UPDATE faces SET embedding=? WHERE id=?", (emb_bytes, face_id))
        conn.commit()
        if face_id in self._cache:
            self._cache[face_id]["embedding"] = embedding.astype(np.float32)
            if thumbnail_bytes is not None:
                self._cache[face_id]["thumbnail"] = thumbnail_bytes

    def find_match(self, embedding, threshold=0.4):
        if not self._cache:
            return None

        best_match = None
        best_score = -1
        emb_norm = np.linalg.norm(embedding)

        for face_id, data in self._cache.items():
            stored_norm = np.linalg.norm(data["embedding"])
            score = float(
                np.dot(embedding, data["embedding"]) / (emb_norm * stored_norm + 1e-6)
            )
            if score > best_score:
                best_score = score
                best_match = {"id": face_id, "name": data["name"], "score": score}

        if best_match and best_match["score"] >= threshold:
            return best_match
        return None

    def find_top_matches(self, embedding, k=5):
        if not self._cache:
            return []

        scores = []
        emb_norm = np.linalg.norm(embedding)

        for face_id, data in self._cache.items():
            stored_norm = np.linalg.norm(data["embedding"])
            score = float(
                np.dot(embedding, data["embedding"]) / (emb_norm * stored_norm + 1e-6)
            )
            scores.append({"id": face_id, "name": data["name"], "score": score})

        scores.sort(key=lambda x: -x["score"])
        return scores[:k]

    def get_all_faces(self):
        return [
            {"id": fid, "name": data["name"]} for fid, data in self._cache.items()
        ]

    def delete_face(self, face_id):
        conn = self._get_conn()
        conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        conn.commit()
        self._cache.pop(face_id, None)

    def get_all_embeddings(self):
        return {fid: data["embedding"] for fid, data in self._cache.items()}

    def get_thumbnails(self, face_ids):
        return {
            fid: self._cache[fid].get("thumbnail")
            for fid in face_ids
            if fid in self._cache
        }

    @property
    def count(self):
        return len(self._cache)

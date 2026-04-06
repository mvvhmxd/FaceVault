import numpy as np
import cv2
import base64
import logging

logger = logging.getLogger(__name__)


class ReconstructionEngine:
    def __init__(self, database):
        self.database = database

    def reconstruct(self, embedding, method="nearest_blend"):
        if method == "pca":
            return self._pca(embedding)
        return self._nearest_blend(embedding)

    def _nearest_blend(self, embedding, k=5):
        all_emb = self.database.get_all_embeddings()
        if not all_emb:
            return self._placeholder(embedding)

        en = np.linalg.norm(embedding)
        sims = {}
        for fid, se in all_emb.items():
            sims[fid] = max(0.0, float(np.dot(embedding, se) / (en * np.linalg.norm(se) + 1e-6)))

        top_ids = sorted(sims, key=sims.get, reverse=True)[:k]
        thumbs = self.database.get_thumbnails(top_ids)

        blended = None
        tw = 0.0
        for fid in top_ids:
            raw = thumbs.get(fid)
            if raw is None:
                continue
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (96, 96))
            w = sims[fid] ** 2
            blended = img.astype(np.float64) * w if blended is None else blended + img.astype(np.float64) * w
            tw += w

        if blended is not None and tw > 0:
            blended = (blended / tw).astype(np.uint8)
            return {
                "image_b64": self._to_b64(blended),
                "similarity": round(max(sims.values()), 3),
                "method": "nearest_blend",
                "num_references": len(top_ids),
            }
        return self._placeholder(embedding)

    def _pca(self, embedding):
        all_emb = self.database.get_all_embeddings()
        if len(all_emb) < 3:
            return self._placeholder(embedding)

        mat = np.array(list(all_emb.values()))
        mean = mat.mean(axis=0)
        U, S, Vt = np.linalg.svd(mat - mean, full_matrices=False)
        nc = min(10, len(S))
        proj = np.dot(embedding - mean, Vt[:nc].T)
        recon = mean + np.dot(proj, Vt[:nc])
        sim = float(np.dot(embedding, recon) / (np.linalg.norm(embedding) * np.linalg.norm(recon) + 1e-6))

        return {
            "image_b64": self._to_b64(self._emb_vis(recon)),
            "similarity": round(sim, 3),
            "method": "pca",
            "explained_variance": round(float(np.sum(S[:nc] ** 2) / (np.sum(S ** 2) + 1e-6)), 3),
        }

    def _placeholder(self, embedding):
        return {
            "image_b64": self._to_b64(self._emb_vis(embedding)),
            "similarity": 0.0,
            "method": "embedding_visualization",
            "num_references": 0,
        }

    def _emb_vis(self, emb):
        norm = (emb - emb.min()) / (emb.max() - emb.min() + 1e-6)
        gs = int(np.ceil(np.sqrt(len(norm))))
        padded = np.zeros(gs * gs)
        padded[: len(norm)] = norm
        grid = padded.reshape(gs, gs)
        vis = cv2.resize(grid.astype(np.float32), (96, 96))
        vis = (vis * 255).astype(np.uint8)
        return cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)

    def interpolate(self, emb1, emb2, steps=5):
        frames = []
        for a in np.linspace(0, 1, steps):
            ie = emb1 * (1 - a) + emb2 * a
            ie /= np.linalg.norm(ie) + 1e-6
            frames.append({"alpha": round(float(a), 2), "image_b64": self._to_b64(self._emb_vis(ie))})
        return frames

    @staticmethod
    def _to_b64(img):
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode()

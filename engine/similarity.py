import numpy as np
from typing import List, Dict
from scipy.spatial.distance import cosine, euclidean


class EmbeddingSimilarity:
    @staticmethod
    def cosine_similarity(emb1, emb2) -> float:
        return float(1 - cosine(np.array(emb1), np.array(emb2)))

    @staticmethod
    def euclidean_distance(emb1, emb2) -> float:
        return float(euclidean(np.array(emb1), np.array(emb2)))

    @staticmethod
    def batch_compare(query, embeddings: list, metric: str = 'cosine') -> List[float]:
        q = np.array(query)
        fn = (lambda a, b: 1 - cosine(a, b)) if metric == 'cosine' else euclidean
        return [float(fn(q, np.array(e))) for e in embeddings]

    @staticmethod
    def normalize(embedding) -> np.ndarray:
        e = np.array(embedding, dtype=np.float64)
        n = np.linalg.norm(e)
        return e / n if n > 1e-8 else e

    @staticmethod
    def compute_drift(embeddings: list) -> Dict:
        if len(embeddings) < 2:
            return {'drift': 0.0, 'stability': 1.0, 'max_deviation': 0.0}
        arrs = [np.array(e) for e in embeddings]
        center = np.mean(arrs, axis=0)
        dists = [float(cosine(e, center)) for e in arrs]
        return {
            'drift': float(np.std(dists)),
            'stability': float(1 - np.mean(dists)),
            'max_deviation': float(np.max(dists)),
        }

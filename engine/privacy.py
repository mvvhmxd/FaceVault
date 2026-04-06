import numpy as np


class PrivacyAnalyzer:
    def __init__(self, reconstruction_engine):
        self.recon = reconstruction_engine

    def analyze(self, embedding):
        result = self.recon.reconstruct(embedding)
        leakage = result.get("similarity", 0.0)
        entropy = self._entropy(embedding)
        uniqueness = self._uniqueness(embedding)
        risk = self._risk_level(leakage)

        return {
            "leakage_score": round(leakage, 3),
            "risk_level": risk,
            "embedding_entropy": round(entropy, 3),
            "uniqueness_score": round(uniqueness, 3),
            "reconstruction_similarity": round(leakage, 3),
            "explanation": self._explain(leakage, risk),
            "recommendations": self._recommend(risk),
        }

    def _entropy(self, emb):
        a = np.abs(emb)
        p = a / (a.sum() + 1e-6)
        ent = -np.sum(p * np.log2(p + 1e-10))
        return float(ent / np.log2(len(emb)))

    def _uniqueness(self, emb):
        all_emb = self.recon.database.get_all_embeddings()
        if not all_emb:
            return 1.0
        en = np.linalg.norm(emb)
        sims = [
            float(np.dot(emb, se) / (en * np.linalg.norm(se) + 1e-6))
            for se in all_emb.values()
        ]
        return 1.0 - max(0.0, min(1.0, float(np.mean(sims))))

    @staticmethod
    def _risk_level(score):
        if score > 0.8:
            return "critical"
        if score > 0.6:
            return "high"
        if score > 0.4:
            return "moderate"
        if score > 0.2:
            return "low"
        return "minimal"

    @staticmethod
    def _explain(score, risk):
        msgs = {
            "critical": "CRITICAL — the embedding can reconstruct a highly similar face. Significant re-identification risk.",
            "high": "HIGH — reconstruction shows strong similarity. Additional privacy protections recommended.",
            "moderate": "MODERATE — some features inferable, but full reconstruction is limited.",
            "low": "LOW — limited reconstruction possible. Privacy exposure is minimal.",
            "minimal": "MINIMAL — embedding is well-protected; reconstruction shows very low similarity.",
        }
        return f"Privacy risk: {risk.upper()} (score {score:.1%}). {msgs.get(risk, '')}"

    @staticmethod
    def _recommend(risk):
        base = [
            "Encrypt embeddings at rest",
            "Implement access controls on the embedding database",
        ]
        if risk in ("critical", "high"):
            base += [
                "Add calibrated noise to stored embeddings",
                "Implement embedding rotation policy",
                "Consider differential-privacy storage",
                "Restrict reconstruction API access",
            ]
        elif risk == "moderate":
            base += [
                "Monitor access patterns",
                "Consider periodic embedding refresh",
            ]
        return base

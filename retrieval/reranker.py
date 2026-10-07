"""Реранкер: cross-encoder для точной пересортировки."""
from sentence_transformers import CrossEncoder

from config import CONFIG
from models import SearchHit


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(CONFIG.reranker_model_name, device=CONFIG.device)

    def rerank(self, query: str, hits: list[SearchHit], top_k: int = 20) -> list[SearchHit]:
        """Пересортировывает кандидатов и возвращает top_k лучших."""
        if not hits:
            return []

        pairs = [(query, h.payload.get("text", "")) for h in hits]
        scores = self.model.predict(pairs)

        for hit, score in zip(hits, scores):
            hit.score = float(score)

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
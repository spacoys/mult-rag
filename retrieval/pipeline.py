"""Retrieval: dense + BM25 → RRF → реранк."""
from collections import defaultdict

from retrieval.searcher import Searcher
from retrieval.reranker import Reranker
from config import CONFIG
from models import SearchHit


class RetrievalPipeline:
    def __init__(self):
        self.searcher = Searcher()
        self.reranker = Reranker()

    def retrieve(self, query: str) -> list[SearchHit]:
        dense = self.searcher.search_dense(query)
        sparse = self.searcher.search_bm25(query)

        print(f"Dense: {len(dense)}")
        print(f"BM25:  {len(sparse)}")

        merged = self._rrf_merge(dense, sparse)
        print(f"RRF:   {len(merged)}")

        reranked = self.reranker.rerank(query, merged, top_k=20)
        unique = self._dedupe_by_page(reranked)[:CONFIG.top_k_rerank]

        print(f"Уникальных страниц: {len(unique)}")
        for h in unique:
            print(f"page={h.payload.get('page')} score={h.score:.3f}")
        return unique

    @staticmethod
    def _rrf_merge(*lists, k: int = 60) -> list[SearchHit]:
        scores = defaultdict(float)
        all_hits: dict[str, SearchHit] = {}
        for lst in lists:
            for rank, hit in enumerate(lst):
                scores[hit.id] += 1.0 / (k + rank + 1)
                all_hits[hit.id] = hit
        merged = sorted(all_hits.values(), key=lambda h: scores[h.id], reverse=True)
        return merged[: CONFIG.top_k_retrieve]

    @staticmethod
    def _dedupe_by_page(hits: list[SearchHit]) -> list[SearchHit]:
        seen = {}
        for h in hits:
            page = h.payload.get("page")
            if page not in seen or h.score > seen[page].score:
                seen[page] = h
        return sorted(seen.values(), key=lambda h: h.score, reverse=True)

    def build_context(self, hits: list[SearchHit]) -> str:
        parts = []
        for h in hits:
            page = h.payload.get("page", "?")
            text = h.payload.get("text", "")
            parts.append(f"[Стр. {page}]\n{text}")
        return "\n\n---\n\n".join(parts)

    def build_prompt(self, query: str, context: str) -> str:
        return (
            "Ответь на вопрос, используя ТОЛЬКО контекст ниже.\n"
            "Если в контексте нет ответа — честно скажи об этом.\n\n"
            f"Контекст:\n{context}\n\n"
            f"Вопрос: {query}\n\nОтвет:"
        )
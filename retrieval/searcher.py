"""Поиск: dense (e5) + BM25."""
from encoders.text import TextEncoder
from encoders.image import ImageEncoder
from storage.qdrant_store import get_store
from storage.bm25_store import BM25Store, tokenize
from config import CONFIG
from models import SearchHit


class Searcher:
    def __init__(self):
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.store = get_store()
        self.bm25 = BM25Store()

    def search_dense(self, query: str):
        query_vec = self.text_encoder.encode_query(query)
        return self.store.search_text(query_vec, top_k=CONFIG.top_k_retrieve)

    def search_bm25(self, query: str):
        self.bm25 = BM25Store()


        tokens = tokenize(query)
        scores = self.bm25.bm25.get_scores(tokens)
        idx_sorted = scores.argsort()[::-1][: CONFIG.top_k_retrieve]

        hits = []
        for i in idx_sorted:
            if scores[i] <= 0:
                continue
            hits.append(SearchHit(
                id=self.bm25.ids[i],
                score=float(scores[i]),
                payload={**self.bm25.metadata[i], "text": self.bm25.corpus[i]},
            ))
        return hits
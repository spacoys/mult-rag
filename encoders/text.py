"""Текстовый энкодер на базе multilingual-e5-small."""
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CONFIG


class TextEncoder:
    def __init__(self):
        self.model = SentenceTransformer(CONFIG.text_model_name, device=CONFIG.device)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"passage: {t}" for t in texts]
        return self.model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            f"query: {query}",
            normalize_embeddings=True,
        )
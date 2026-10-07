"""Все настройки проекта в одном месте."""
from dataclasses import dataclass


@dataclass
class Config:
    text_model_name: str = "intfloat/multilingual-e5-base"
    image_model_name: str = "ViT-B-32"
    image_pretrained: str = "laion2b_s34b_b79k"
    reranker_model_name: str = "DiTy/cross-encoder-russian-msmarco"

    chunk_size: int = 1500
    chunk_overlap: int = 200

    qdrant_path: str = "./qdrant_data"
    collection_name: str = "documents"
    text_vector_size: int = 768          
    image_vector_size: int = 512 

    top_k_retrieve: int = 100
    top_k_rerank: int = 10

    device: str = "cpu"


CONFIG = Config()
"""Датаклассы — что мы передаём между модулями.

Вместо словарей с непонятными ключами — типизированные объекты.
"""
from dataclasses import dataclass, field
from typing import Literal
import numpy as np


@dataclass
class TextChunk:
    """Кусок текста из документа."""
    text: str
    page: int
    chunk_id: str  
    modality: str = "text"


@dataclass
class ImageItem:
    """Картинка из документа."""
    raw_bytes: bytes
    page: int
    image_id: str
    modality: str = "image"


@dataclass
class SearchHit:
    """Один результат поиска."""
    id: str
    score: float
    payload: dict


@dataclass
class RetrievalResult:
    """Итог поиска: тексты + картинки, готовые для LLM."""
    texts: list[SearchHit] = field(default_factory=list)
    images: list[SearchHit] = field(default_factory=list)
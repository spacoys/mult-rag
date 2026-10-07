"""Извлечение контента из PDF.

Отвечает ТОЛЬКО за чтение PDF. Никаких эмбеддингов, Qdrant и прочего.
"""
import fitz  # PyMuPDF
import hashlib
from pathlib import Path

from models import TextChunk, ImageItem


def extract_pdf(pdf_path: str) -> tuple[list[TextChunk], list[ImageItem]]:
    """
    Читает PDF и возвращает список текстовых чанков и картинок.
    Чанкинг здесь примитивный (по страницам) — точный чанкинг в chunker.py.
    """
    doc = fitz.open(pdf_path)
    texts: list[TextChunk] = []
    images: list[ImageItem] = []

    for page_num, page in enumerate(doc, start=1):
        # --- Текст страницы ---
        page_text = page.get_text("text").strip()
        if page_text:
            texts.append(TextChunk(
                text=page_text,
                page=page_num,
                chunk_id=_make_id(f"text_{page_num}_{page_text[:50]}"),
            ))

        # --- Картинки страницы ---
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            images.append(ImageItem(
                raw_bytes=base["image"],
                page=page_num,
                image_id=_make_id(f"img_{page_num}_{img_index}"),
            ))

    doc.close()
    return texts, images


def _make_id(seed: str) -> str:
    """Детерминированный ID: одинаковый вход → одинаковый ID."""
    return hashlib.md5(seed.encode()).hexdigest()[:16]
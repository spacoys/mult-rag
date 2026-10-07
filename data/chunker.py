"""Разбиение текста на чанки по границам предложений.

Это критично: если резать по символам, слова рвутся и поиск ухудшается.
"""
import re
from models import TextChunk
from config import CONFIG


def chunk_text(text: str, page: int, chunk_id_prefix: str) -> list[TextChunk]:
    """
    Режет текст на чанки по границам предложений.
    Использует простую стратегию: сначала абзацы, потом предложения.
    """
    # Сначала разбиваем по абзацам (пустая строка)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[TextChunk] = []
    current = ""

    for para in paragraphs:
        # Если абзац сам больше chunk_size — режем по предложениям
        if len(para) > CONFIG.chunk_size:
            for sentence in _split_sentences(para):
                if len(current) + len(sentence) > CONFIG.chunk_size:
                    if current:
                        chunks.append(_make_chunk(current, page, chunk_id_prefix, len(chunks)))
                    current = sentence
                else:
                    current += " " + sentence
        else:
            if len(current) + len(para) > CONFIG.chunk_size:
                if current:
                    chunks.append(_make_chunk(current, page, chunk_id_prefix, len(chunks)))
                current = para
            else:
                current = (current + "\n\n" + para).strip()

    if current:
        chunks.append(_make_chunk(current, page, chunk_id_prefix, len(chunks)))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Простое разбиение по . ! ?"""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _make_chunk(text: str, page: int, prefix: str, index: int) -> TextChunk:
    return TextChunk(
        text=text,
        page=page,
        chunk_id=f"{prefix}_{index}",
    )
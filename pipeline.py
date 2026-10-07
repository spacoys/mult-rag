"""Индексация: PDF → чанки → векторы → Qdrant + BM25."""
from data.extractor import extract_pdf
from data.chunker import chunk_text
from encoders.text import TextEncoder
from encoders.image import ImageEncoder
from storage.qdrant_store import get_store
from storage.bm25_store import BM25Store


def index_pdf(pdf_path: str):
    raw_texts, images = extract_pdf(pdf_path)

    chunks = []
    for t in raw_texts:
        chunks.extend(chunk_text(t.text, t.page, t.chunk_id))

    text_encoder = TextEncoder()
    text_vectors = text_encoder.encode_documents([c.text for c in chunks])

    image_vectors = None
    if images:
        image_encoder = ImageEncoder()
        image_vectors = image_encoder.encode_images([i.raw_bytes for i in images])

    store = get_store()

    store.upsert_texts(
        ids=[c.chunk_id for c in chunks],
        vectors=text_vectors,
        payloads=[{"text": c.text, "page": c.page, "modality": "text"} for c in chunks],
    )

    if images and image_vectors is not None and len(image_vectors) > 0:
        store.upsert_images(
            ids=[i.image_id for i in images],
            vectors=image_vectors,
            payloads=[{"page": i.page, "modality": "image"} for i in images],
        )

    bm25_store = BM25Store()
    bm25_store.build(
        texts=[c.text for c in chunks],
        ids=[c.chunk_id for c in chunks],
        metadata=[{"page": c.page, "modality": "text"} for c in chunks],
    )

    print(f"Готово: {len(chunks)} чанков, {len(images)} картинок")
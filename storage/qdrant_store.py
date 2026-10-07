"""Работа с Qdrant: синглтон + UUID для ID."""
import uuid

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from config import CONFIG
from models import SearchHit


NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")
_store_instance = None


def get_store() -> "QdrantStore":
    global _store_instance
    if _store_instance is None:
        _store_instance = QdrantStore()
    return _store_instance


def to_uuid(string_id: str) -> str:
    """Детерминированно превращает строку в UUID."""
    return str(uuid.uuid5(NAMESPACE, string_id))


class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(path=CONFIG.qdrant_path)
        self.collection = CONFIG.collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "text": qm.VectorParams(
                    size=CONFIG.text_vector_size,
                    distance=qm.Distance.COSINE,
                ),
                "image": qm.VectorParams(
                    size=CONFIG.image_vector_size,
                    distance=qm.Distance.COSINE,
                ),
            },
        )

    def upsert_texts(self, ids, vectors, payloads):
        points = [
            qm.PointStruct(id=to_uuid(i), vector={"text": v.tolist()}, payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def upsert_images(self, ids, vectors, payloads):
        points = [
            qm.PointStruct(id=to_uuid(i), vector={"image": v.tolist()}, payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search_text(self, query_vector, top_k: int):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            using="text",
            limit=top_k,
            with_payload=True,
        )
        return [SearchHit(id=str(p.id), score=p.score, payload=p.payload) for p in results.points]

    def search_image(self, query_vector, top_k: int):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            using="image",
            limit=top_k,
            with_payload=True,
        )
        return [SearchHit(id=str(p.id), score=p.score, payload=p.payload) for p in results.points]
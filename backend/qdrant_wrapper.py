from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid


class QdrantWrapper:
    def __init__(self, url="http://qdrant:6333", collection="pdf_docs", dim=768):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.dim = dim

    def create_collection(self):
        """Create or reset a collection in Qdrant."""
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.dim,
                distance=Distance.COSINE
            )
        )

    def insert_chunks(self, chunks, embeddings):
        """Insert text chunks + embeddings into Qdrant."""
        points = []
        for text, vector in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"text": text}
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, embedding, top_k=3):
        """Retrieve similar vectors."""
        results = self.client.search(
            collection_name=self.collection,
            query_vector=embedding,
            limit=top_k
        )
        return [result.payload["text"] for result in results]

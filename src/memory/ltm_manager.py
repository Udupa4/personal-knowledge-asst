import os
import logging
from datetime import datetime, timezone
from typing import List
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue, PayloadSchemaType

from src.qa.retriever import get_qdrant_client
from src.utils.embeddings import select_embeddings
from src.utils.singletone import SingletonMeta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LTM_PERSIST_DIR = os.environ.get("LTM_PERSIST_DIR", "./ltm_store")
LTM_COLLECTION = "ltm"
VECTOR_SIZE = 3072 if os.environ.get("EMBEDDING_PROVIDER", "") == "gemini" else 1024

class LtmManager(metaclass=SingletonMeta):
    def __init__(self):
        self.embeddings = select_embeddings()
        client = get_qdrant_client()
        # Create collection if it doesn't exist
        existing = [c.name for c in client.get_collections().collections]
        if LTM_COLLECTION not in existing:
            from qdrant_client.models import Distance, VectorParams
            client.create_collection(
                collection_name=LTM_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
        client.create_payload_index(
            collection_name=LTM_COLLECTION,
            field_name="metadata.user_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self.ltm_store = QdrantVectorStore(
            client=client,
            collection_name=LTM_COLLECTION,
            embedding=self.embeddings,
        )

    def save(self, summary: str, user_id: str, session_id: str):
        """Save the summary to the ltm store"""
        self.ltm_store.add_texts(
            texts=[summary],
            metadatas=[{
                "user_id": user_id,
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }]
        )
        logger.info(f"LTM saved for user_id='{user_id}', session_id='{session_id}'")

    def retrieve(self, query: str, user_id: str, k: int = 3) -> List[str]:
        qdrant_filter = Filter(
            must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))]
        )
        results = self.ltm_store.similarity_search(
            query, k=k, filter=qdrant_filter
        )
        return [r.page_content for r in results]

    @staticmethod
    def get_all_for_user(user_id: str) -> list:
        client = get_qdrant_client()
        results, _ = client.scroll(
            collection_name=LTM_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))]
            ),
            with_payload=True,
            limit=100,
        )
        return [
            {
                "id": str(r.id),
                "summary": r.payload.get("page_content", ""),
                "session_id": r.payload.get("metadata", {}).get("session_id"),
                "created_at": r.payload.get("metadata", {}).get("created_at"),
            }
            for r in results
        ]

    @staticmethod
    def delete_for_user(user_id: str):
        client = get_qdrant_client()
        from qdrant_client.models import FilterSelector
        client.delete(
            collection_name=LTM_COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))]
                )
            )
        )

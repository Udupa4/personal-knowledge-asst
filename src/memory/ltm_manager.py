import os
import logging
from typing import List
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

from src.common.utils.embeddings import select_embeddings
from src.common.utils.singletone import SingletonMeta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LTM_PERSIST_DIR = os.environ.get("LTM_PERSIST_DIR", "./ltm_store")

class LtmManager(metaclass=SingletonMeta):
    def __init__(self):
        self.embeddings = select_embeddings()
        self.ltm_store = Chroma(
            collection_name="ltm",
            embedding_function=self.embeddings,
            persist_directory=LTM_PERSIST_DIR
        )

    def save(self, summary: str, user_id: str, session_id: str):
        """Save the summary to the ltm store"""
        self.ltm_store.add_texts(
            texts=[summary],
            metadatas=[{"user_id": user_id, "session_id": session_id}]
        )
        logger.info(f"LTM saved for user_id='{user_id}', session_id='{session_id}'")

    def retrieve(self, query: str, user_id: str, k: int = 3) -> List[str]:
        # Use explicit $eq operator — shorthand dict filter is unreliable in ChromaDB
        where_filter = {"user_id": {"$eq": user_id}}

        # Guard: check how many docs exist for this user before querying
        try:
            existing = self.ltm_store._collection.get(where=where_filter)
            available = len(existing["ids"])
        except Exception as ex:
            logger.error(f"Failed to retrieve available memory for user {user_id}. {ex}")
            available = 0

        if available == 0:
            return []

        safe_k = min(k, available)  # never request more than what exists

        results = self.ltm_store.similarity_search(
            query, k=safe_k, filter=where_filter
        )
        return [r.page_content for r in results]

    def get_all(self) -> dict:
        """Fetch all documents from LTM store, grouped by user_id."""
        raw = self.ltm_store._collection.get(include=["documents", "metadatas"])
        grouped = {}
        for doc, meta in zip(raw["documents"], raw["metadatas"]):
            uid = meta.get("user_id", "unknown")
            grouped.setdefault(uid, []).append({
                "summary": doc,
                "session_id": meta.get("session_id"),
            })
        return grouped

    def get_all_for_user(self, user_id: str) -> list:
        """Fetch all LTM documents for a specific user."""
        raw = self.ltm_store._collection.get(
            where={"user_id": {"$eq": user_id}},
            include=["documents", "metadatas"]
        )
        return [
            {"id": i, "summary": d, "session_id": m.get("session_id")}
            for i, d, m in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]

    def delete_for_user(self, user_id: str):
        """Delete all LTM entries for a given user."""
        raw = self.ltm_store._collection.get(where={"user_id": {"$eq": user_id}})
        if raw["ids"]:
            self.ltm_store._collection.delete(ids=raw["ids"])

import os
import logging
import json
import hashlib
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.storage import RedisStore
from langchain_classic.storage._lc_store import create_kv_docstore

from src.utils.embeddings import select_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception as e:
    logger.warning(f"Failed to import GoogleGenerativeAIEmbeddings: {e}")
    GoogleGenerativeAIEmbeddings = None

DATA_DIR = "./data/docs"
COLLECTION_PREFIX = "rag"
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_store")
DEFAULT_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
GOOGLE_API_KEY = None
MANIFEST_PATH = "./data/.ingest_manifest.json"

def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.environ.get("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY"),
    )

_qdrant_client: Optional[QdrantClient] = None

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = _get_qdrant_client()
    return _qdrant_client

class ChunkedDocLoader:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # --- Manifest helpers ---
    @staticmethod
    def _load_manifest() -> dict:
        """Load the ingest manifest tracking which files have been ingested."""
        if os.path.exists(MANIFEST_PATH):
            logger.info(f"Loading manifest from: {MANIFEST_PATH}")
            with open(MANIFEST_PATH, "r") as f:
                return json.load(f)
        logger.info(f"Manifest not found in path: {MANIFEST_PATH}")
        return {}

    @staticmethod
    def _save_manifest(manifest: dict):
        """Persist updated manifest to disk."""
        os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)

    @staticmethod
    def _file_hash(path: str) -> str:
        """Compute MD5 hash of a file to detect new or modified files."""
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    # --- Core load ---
    def load_and_split(
        self,
        doc_dir: Optional[str] = None,
        only_new: bool = False
    ) -> List[Document]:
        """
        Load .txt files from doc_dir and return full Documents (not pre-chunked).
        Chunking is delegated to ParentDocumentRetriever internally.

        Args:
            doc_dir:  Override the default DATA_DIR.
            only_new: Skip files already recorded in the manifest (by MD5).
                      Saves the updated manifest on completion.

        Returns:
            List of full Documents (one per file), not yet split into chunks.
        """
        base = doc_dir or DATA_DIR
        manifest = self._load_manifest() if only_new else {}
        updated_manifest = dict(manifest)
        docs: List[Document] = []

        if not os.path.exists(base):
            os.makedirs(base, exist_ok=True)
            logger.info(f"Created missing doc dir: {base}")

        for fname in sorted(os.listdir(base)):
            if not fname.lower().endswith(".txt"):
                continue

            path = os.path.join(base, fname)
            file_hash = self._file_hash(path)

            if only_new and manifest.get(fname) == file_hash:
                logger.info(f"Skipping already-ingested file: {fname}")
                continue

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            title = next(
                (line.strip() for line in text.splitlines() if line.strip()),
                fname
            )

            # Return the full document — ParentDocumentRetriever will handle splitting
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": path,
                    "title": title,
                    "filename": fname,
                }
            ))
            updated_manifest[fname] = file_hash
            logger.info(f"Loaded '{fname}'")

        if only_new:
            self._save_manifest(updated_manifest)

        logger.info(f"Total new documents to ingest: {len(docs)}")
        return docs

class VectorRetriever:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.collection_name = f"{COLLECTION_PREFIX}_{user_id}"
        self.embeddings = select_embeddings()
        self.docstore: RedisStore = self._build_docstore()
        self.vectordb: Optional[QdrantVectorStore] = None
        self.parent_retriever: Optional[ParentDocumentRetriever] = None

        # Child splitter: small chunks → precise embedding & retrieval
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        # Parent splitter: large chunks → rich context passed to LLM
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200
        )

    # --- Docstore persistence for parent chunks in Redis ---
    def _build_docstore(self):
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        byte_store = RedisStore(
            redis_url=redis_url,
            namespace=f"docstore:{self.user_id}",
            ttl=None,
        )
        return create_kv_docstore(byte_store)

    def _collection_exists(self) -> bool:
        client = get_qdrant_client()
        existing = [c.name for c in client.get_collections().collections]
        return self.collection_name in existing

    def _build_vectordb(self) -> QdrantVectorStore:
        client = get_qdrant_client()
        if not self._collection_exists():
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,  # Gemini embedding-001 outputs 768 dimensional embedding
                    distance=Distance.COSINE,
                )
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
        return QdrantVectorStore(
            client=client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def _build_parent_retriever(self) -> ParentDocumentRetriever:
        """Construct the ParentDocumentRetriever from current vectordb + docstore."""
        return ParentDocumentRetriever(
            vectorstore=self.vectordb,
            docstore=self.docstore,
            child_splitter=self._child_splitter,
            parent_splitter=self._parent_splitter,
        )

    def build_or_load(self, docs: List[Document], add_new: bool = False):
        self.vectordb = self._build_vectordb()  # creates collection if not exists
        self.parent_retriever = self._build_parent_retriever()

        if docs and add_new:
            logger.info(f"Ingesting {len(docs)} docs for user {self.user_id}")
            self.parent_retriever.add_documents(docs)

    # --- Retrieval ---
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant parent chunks for a query.

        Flow:
          1. Child chunks are searched in QdrantDB by vector similarity
          2. Their parent chunks are fetched from the RedisStore
          3. Parent chunks (large, full context) are returned to the LLM

        Returns a list of dicts with title, filename, path, and full content.
        """
        if not self.parent_retriever:
            logger.warning("VectorRetriever: parent_retriever is not initialized.")
            return []

        # ParentDocumentRetriever.invoke() returns parent Documents directly
        # search_kwargs controls how many children are searched in QdrantDB
        self.parent_retriever.search_kwargs = {"k": top_k * 2}  # fetch more children to get top_k parents
        parent_docs = self.parent_retriever.invoke(query)

        # Deduplicate by source in case multiple children map to same parent
        seen_sources = set()
        out = []
        for doc in parent_docs:
            source = doc.metadata.get("source", "")
            if source in seen_sources:
                continue
            seen_sources.add(source)
            out.append({
                "title": doc.metadata.get("title"),
                "filename": doc.metadata.get("filename"),
                "path": source,
                "snippet": doc.page_content.strip(),  # full parent chunk — rich context
            })
            if len(out) >= top_k:
                break

        return out
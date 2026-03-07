import os
import logging
import pickle
import json
import hashlib
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore

from src.common.utils.embeddings import select_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception as e:
    logger.warning(f"Failed to import GoogleGenerativeAIEmbeddings: {e}")
    GoogleGenerativeAIEmbeddings = None

DATA_DIR = "./data/docs"
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_store")
DEFAULT_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
GOOGLE_API_KEY = None
MANIFEST_PATH = "./data/.ingest_manifest.json"
DOCSTORE_PATH = "./data/.docstore.pkl"     # persisted parent docstore

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
            with open(MANIFEST_PATH, "r") as f:
                return json.load(f)
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
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        self.vectordb: Optional[Chroma] = None
        self.embeddings = select_embeddings()
        self.docstore: InMemoryStore = self._load_docstore()
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

    # --- Docstore persistence ---
    @staticmethod
    def _load_docstore() -> InMemoryStore:
        """Load persisted parent docstore from disk if it exists."""
        if os.path.exists(DOCSTORE_PATH):
            try:
                with open(DOCSTORE_PATH, "rb") as f:
                    store = pickle.load(f)
                logger.info("Loaded parent docstore from disk.")
                return store
            except Exception as ex:
                logger.warning(f"Failed to load docstore, starting fresh: {ex}")
        return InMemoryStore()

    def _save_docstore(self):
        """Persist the in-memory parent docstore to disk."""
        os.makedirs(os.path.dirname(DOCSTORE_PATH), exist_ok=True)
        with open(DOCSTORE_PATH, "wb") as f:
            pickle.dump(self.docstore, f)
        logger.info("Saved parent docstore to disk.")

    # --- Chroma helpers ---
    def _chroma_exists(self) -> bool:
        return os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3"))

    def _build_parent_retriever(self) -> ParentDocumentRetriever:
        """Construct the ParentDocumentRetriever from current vectordb + docstore."""
        return ParentDocumentRetriever(
            vectorstore=self.vectordb,
            docstore=self.docstore,
            child_splitter=self._child_splitter,
            parent_splitter=self._parent_splitter,
        )

    # --- Build / Load ---
    def build_or_load(self, docs: List[Document], add_new: bool = False):
        """
        Load an existing Chroma store or create one from full documents.

        The ParentDocumentRetriever handles splitting internally:
          - Small child chunks are embedded → stored in ChromaDB
          - Large parent chunks are stored in docstore (persisted via pickle)

        Args:
            docs:    Full Documents (one per file) from ChunkedDocLoader.
            add_new: If True and store exists, add only new docs.
                     If False, load existing store without modification.
        """
        os.makedirs(self.persist_directory, exist_ok=True)

        if self._chroma_exists():
            logger.info("Found existing Chroma store — loading from disk.")
            self.vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            self.parent_retriever = self._build_parent_retriever()

            if add_new and docs:
                logger.info(f"Adding {len(docs)} new documents to existing store.")
                self.parent_retriever.add_documents(docs)
                self._save_docstore()
                logger.info("New documents added and docstore saved.")
            elif not docs:
                logger.info("No new documents to add.")
        else:
            if not docs:
                logger.warning("No docs provided and no existing Chroma store found.")
                return
            logger.info(f"Creating new Chroma store from {len(docs)} documents.")
            self.vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            self.parent_retriever = self._build_parent_retriever()
            self.parent_retriever.add_documents(docs)
            self._save_docstore()
            logger.info("Chroma store and docstore created and saved.")

    # --- Retrieval ---
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant parent chunks for a query.

        Flow:
          1. Child chunks are searched in ChromaDB by vector similarity
          2. Their parent chunks are fetched from the docstore
          3. Parent chunks (large, full context) are returned to the LLM

        Returns a list of dicts with title, filename, path, and full content.
        """
        if not self.parent_retriever:
            logger.warning("VectorRetriever: parent_retriever is not initialized.")
            return []

        # ParentDocumentRetriever.invoke() returns parent Documents directly
        # search_kwargs controls how many children are searched in ChromaDB
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
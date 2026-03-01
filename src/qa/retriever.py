import os
import logging
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import SecretStr

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

class ChunkedDocLoader:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_and_split(self, doc_dir: Optional[str] = None) -> List[Document]:
        base = doc_dir or DATA_DIR
        logger.info(f"Loading and Splitting documents from {base}")
        docs = []
        if not os.path.exists(base):
            os.makedirs(base, exist_ok=True)
        for fname in sorted(os.listdir(base)):
            if not fname.lower().endswith(".txt"):
                continue
            path = os.path.join(base, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            title = text.splitlines()[0].strip() if text.splitlines() else fname
            full_doc = Document(page_content=text, metadata={"source": path, "title": title})
            chunks = self.splitter.split_documents([full_doc])
            docs.extend(chunks)
        logger.info(f"Loaded and split {len(docs)} documents using RecursiveSplitter")
        return docs

class VectorRetriever:
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        self.vectordb: Optional[Chroma] = None
        self.embeddings = self._select_embeddings()

    @staticmethod
    def _select_embeddings():
        global GOOGLE_API_KEY
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
        provider = os.environ.get("EMBEDDING_PROVIDER", "huggingface").lower()
        if provider in ("google", "gemini", "google_genai") and GOOGLE_API_KEY:
            logger.info("Using Google Generative AI for embeddings")
            try:
                return GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL, api_key=SecretStr(GOOGLE_API_KEY))
            except Exception as ex:
                logger.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {ex}")
                from langchain_huggingface import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            # Local HuggingFace embeddings (no external key)
            logger.info("Using HuggingFace for embeddings")
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def build_or_load(self, docs: List[Document]):
        os.makedirs(self.persist_directory, exist_ok=True)
        if os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
            logger.info("Found existing Chroma store. Using the same as persist_directory")
            self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        else:
            logger.info(f"Creating new Chroma store from {DATA_DIR}")
            self.vectordb = Chroma.from_documents(docs, self.embeddings, persist_directory=self.persist_directory)
            logger.info(f"Created Chroma store out of {len(docs)} documents")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.vectordb:
            return []
        results = self.vectordb.similarity_search_with_score(query, k=top_k)
        out = []
        for doc, raw_score in results:
            out.append({
                "title": doc.metadata.get("title"),
                "path": doc.metadata.get("source"),
                "distance": float(raw_score),
                "snippet": (doc.page_content[:800].strip() if doc.page_content else "")
            })
        out_sorted = sorted(out, key=lambda x: x["distance"])   # sort by distance, lower is better
        return out_sorted
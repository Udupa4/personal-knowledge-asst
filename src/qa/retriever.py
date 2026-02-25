import os
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DATA_DIR = "./data/docs"

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_store")

class ChunkedDocLoader:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_and_split(self, doc_dir: Optional[str] = None) -> List[Document]:
        """
        Load and split documents into chunks. Accepts only .txt files for now.
        """
        base = doc_dir or DATA_DIR
        docs = []
        for fname in sorted(os.listdir(base)):
            if not fname.lower().endswith(".txt"):
                continue
            path = os.path.join(base, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            # title heuristics: first non-empty line
            title = text.splitlines()[0].strip() if text.splitlines() else fname
            # create Document with metadata for provenance
            full_doc = Document(page_content=text, metadata={"source": path, "title": title})
            # split into chunks
            chunks = self.splitter.split_documents([full_doc])
            docs.extend(chunks)
        return docs

class VectorRetriever:
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        self.vectordb = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'local_files_only': True}
        )

    def build_or_load(self, docs: List[Document]):
        """
        Create or update the local Chroma store from a list of Documents (chunks).
        """
        os.makedirs(self.persist_directory, exist_ok=True)
        if os.path.exists(os.path.join(self.persist_directory, "index.sqlite")):
            # naive check; load persistent store
            self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            # Upsert new docs if any (we'll upsert to allow updates)
            if docs:
                self.vectordb.add_documents(docs)
        else:
            # create new store
            self.vectordb = Chroma.from_documents(docs, self.embeddings, persist_directory=self.persist_directory)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.vectordb:
            return []
        results = self.vectordb.similarity_search_with_score(query, k=top_k)    # results: list of (Document, score)
        out = []
        for doc, score in results:
            out.append({
                "title": doc.metadata.get("title"),
                "path": doc.metadata.get("source"),
                "score": float(score),
                "snippet": doc.page_content[:500].strip()
            })
        return out
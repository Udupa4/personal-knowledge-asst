import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_function = None

def initialize_embedding_function():
    global embedding_function
    google_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    provider = os.environ.get("EMBEDDING_PROVIDER", "huggingface").lower()
    default_embedding_model = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    if provider in ("google", "gemini", "google_genai") and google_api_key:
        logger.info("Using Google Generative AI for embeddings")
        try:
            embedding_function = GoogleGenerativeAIEmbeddings(model=default_embedding_model, api_key=SecretStr(google_api_key))
        except Exception as ex:
            logger.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {ex}")
            from langchain_huggingface import HuggingFaceEmbeddings
            embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        logger.info("Using HuggingFace for embeddings")
        from langchain_huggingface import HuggingFaceEmbeddings
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def select_embeddings():
    return embedding_function

initialize_embedding_function()
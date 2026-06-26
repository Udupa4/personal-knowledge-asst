import os
import logging
from pydantic import SecretStr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_function = None


def initialize_embedding_function():
    global embedding_function

    google_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    provider = os.environ.get("EMBEDDING_PROVIDER", "huggingface").lower()
    default_embedding_model = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

    # Define our upgraded local configuration
    # Note: Using the v1.5 BGE model for top-tier local semantic accuracy
    local_model_name = "BAAI/bge-large-en-v1.5"
    local_model_kwargs = {"device": "cuda"}  # Change to "cuda" if running on a GPU machine
    local_encode_kwargs = {"normalize_embeddings": True}
    local_query_kwargs = {"normalize_embeddings": True, "prompt": "query: "}

    if provider in ("google", "gemini", "google_genai") and google_api_key:
        logger.info("Using Google Generative AI for embeddings")
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embedding_function = GoogleGenerativeAIEmbeddings(
                model=default_embedding_model,
                api_key=SecretStr(google_api_key)
            )
        except Exception as ex:
            logger.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {ex}. Falling back to Local.")
            from langchain_huggingface import HuggingFaceEmbeddings
            embedding_function = HuggingFaceEmbeddings(
                model_name=local_model_name,
                model_kwargs=local_model_kwargs,
                encode_kwargs=local_encode_kwargs,
                query_encode_kwargs=local_query_kwargs
            )
    else:
        logger.info(f"Using HuggingFace local embeddings ({local_model_name})")
        from langchain_huggingface import HuggingFaceEmbeddings
        embedding_function = HuggingFaceEmbeddings(
            model_name=local_model_name,
            model_kwargs=local_model_kwargs,
            encode_kwargs=local_encode_kwargs,
            query_encode_kwargs=local_query_kwargs
        )

def select_embeddings():
    return embedding_function

initialize_embedding_function()
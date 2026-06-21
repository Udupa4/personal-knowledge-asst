import logging
import os
from pydantic import SecretStr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "google").lower()

# Google
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b")

DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
DEFAULT_TEMPERATURE = 0.7


def get_llm(temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS):
    logger.info(f"Initializing LLM provider: {LLM_PROVIDER}")
    if LLM_PROVIDER == "ollama":
        return _get_ollama_llm(temperature, max_tokens)
    return _get_google_llm(temperature, max_tokens)


def _get_google_llm(temperature: float, max_tokens: int):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Run: pip install langchain-google-genai")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set.")
    return ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        api_key=SecretStr(GOOGLE_API_KEY),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _get_ollama_llm(temperature: float, max_tokens: int):
    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Run: pip install langchain-openai")
    return ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        api_key=SecretStr("ollama"),
        temperature=temperature,
        max_tokens=max_tokens,
    )

import os
from pydantic import SecretStr

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ModuleNotFoundError:
    ChatGoogleGenerativeAI = None

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
DEFAULT_MAX_TOKENS = int(os.environ.get("GEMINI_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = 0.0

def get_llm(model: str = None, api_key: str = GOOGLE_API_KEY,
            temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS):
    """
    Return a configured ChatGoogleGenerativeAI langchain instance
    :param model: Gemini model to be used
    :param api_key: api_key to be used, default picked from .env file
    :param temperature: Defaults to 0.0 for consistency.
    :param max_tokens: Defaults to 512.
    :return: ChatGoogleGenerativeAI langchain instance.
    """
    if ChatGoogleGenerativeAI is None:
        raise ModuleNotFoundError("langchain-google-genai is not installed. Run 'pip install langchain-google-genai'")
    model = model or DEFAULT_MODEL
    api_key = api_key or GOOGLE_API_KEY
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set. Please Configure it.")
    temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

    llm = ChatGoogleGenerativeAI(model=model, api_key=SecretStr(api_key), temperature=temperature, max_tokens=max_tokens)
    return llm

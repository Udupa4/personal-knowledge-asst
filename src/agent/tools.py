import logging
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langchain_community.tools.tavily_search import TavilySearchResults
from dataclasses import dataclass

from src.memory.ltm_manager import LtmManager
from src.qa.retriever import VectorRetriever, ChunkedDocLoader

logger = logging.getLogger(__name__)

# Initialise ltm manager
_ltm = LtmManager()

# Initialize retriever
_loader = ChunkedDocLoader(chunk_size=800, chunk_overlap=200)
_retriever = VectorRetriever()
_docs = _loader.load_and_split()
_retriever.build_or_load(_docs)

tavily = TavilySearchResults(max_results=3)

@dataclass
class UserContext:
    user_id: str

@tool
def search_knowledge_base(query: str) -> str:
    """
        Search the personal knowledge base for factual information. Use this when the question requires information
        from stored documents.
        Input must be a focused search query, not the full user question.
        Returns:
            most relevant document snippets found. If no relevant documents are found, returns a not-found message.
    """
    try:
        docs = _retriever.retrieve(query)
        if not docs:
            return f"No relevant documents found for the query."

        parts = []
        for doc in docs:
            parts.append(f"[{doc['filename']}]\n{doc['snippet']}")

        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"search_knowledge_base failed: {e}")
        return f"Knowledge base search failed: {e}"

@tool
def recall_user_memory(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """
    Retrieve relevant long-term memories about the user from past sessions.
    Use this when the question refers to past conversations, previously discussed topics,
    or user preferences and history.
    Input should be a query describing what you want to remember about the user.
    """
    try:
        user_id = runtime.context.user_id
        memories = _ltm.retrieve(query, user_id, k=3)
        if not memories:
            return "No relevant memories found for this user."
        return "\n".join(f"- {m}" for m in memories)
    except Exception as e:
        logger.error(f"recall_user_memory failed: {e}")
        return f"Memory recall failed: {e}"

@tool
def web_search(query: str) -> str:
    """
    Search the web for current or general information not found in the knowledge base.
    Use this only after search_knowledge_base returns no relevant results,
    or when the question explicitly requires up-to-date information.
    Input should be a clear, concise web search query.
    Returns:
        A maximum of 3 results each containing url and content.
    """
    try:
        results = tavily.invoke({"query": query})
        if not results:
            return "No web search results found."
        parts = []
        for r in results:
            parts.append(f"[{r.get('url', 'web')}]\n{r.get('content', '')}")
        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"web_search failed: {e}")
        return f"Web search failed: {e}"


def get_tools() -> list:
    """
    Assemble the full tool list for a given user_id.
    Called by agent_node at the start of each graph invocation.

    Returns tools in priority order — the LLM tends to try tools
    listed earlier first when multiple seem applicable.
    """
    return [
        search_knowledge_base,
        recall_user_memory,
        web_search,
    ]
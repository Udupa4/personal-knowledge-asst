import asyncio
import logging

from src.qa.retriever import ChunkedDocLoader, VectorRetriever
from src.agent.state import AgentState

logger = logging.getLogger(__name__)

# initialize retriever and loader
_loader = ChunkedDocLoader(chunk_size=800, chunk_overlap=200)
_retriever = VectorRetriever()
_docs = _loader.load_and_split()
_retriever.build_or_load(_docs)

async def retrieve_node(state: AgentState) -> dict:
    """
    Retrieve top-k relevant parent chunks from the vector store.
    Only runs when route is "rag" or "rag_and_memory". This node is never called for "memory_only" or "clarify" routes.

    Returns:
        retrieved_docs: list of dicts with title, filename, path, snippet
    """
    try:
        # Wrap retrieve function in to_thread since it's a sync call.
        retrieved_docs = await asyncio.to_thread(
            _retriever.retrieve,
            state["question"],
            state["top_k"]
        )

        logger.info(f"Retrieved {len(retrieved_docs)} matching documents.")
        return {"retrieved_docs": retrieved_docs}
    except Exception as e:
        logger.error(f"Error while trying to fetch matching docs in retrieve node. {e}")
        return {"retrieved_docs": [], "error": str(e)}
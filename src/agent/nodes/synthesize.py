import logging

from src.qa.answerer import compose_prompt, synthesize_answer
from src.agent.state import AgentState

logger = logging.getLogger(__name__)

async def synthesize_node(state: AgentState) -> dict:
    """
    Compose a prompt from all available context and call the LLM to generate an answer.

    Works for three routes:
    - "rag":            retrieved_docs populated, memory may be empty
    - "memory_only":    retrieved_docs is [], stm/ltm context drives the answer
    - "rag_and_memory": both retrieved_docs and memory context are populated

    The compose_prompt function handles empty retrieved_docs gracefully
    (it renders "No Evidence Found."), so no branching is needed here.

    Returns:
        answer:   the LLM's response string
        metadata: token counts, model name, finish reason
    """
    try:
        prompt = compose_prompt(
            stm_context=state.get("stm_context", []),
            ltm_context=state.get("ltm_context", []),
            retrieved=state.get("retrieved_docs", []),
            user_question=state["question"]
        )

        answer, metadata = await synthesize_answer(prompt)

        logger.info(f"synthesize_node answer length: {len(answer)}. "
                    f"Token used: {metadata.get('usage', 'None')}")

        return {"answer": answer, "metadata": metadata}
    except Exception as e:
        logger.error(f"Error while executing synthesize node. {e}")
        return {"answer": "Encountered an error while generating a response.", "metadata": {}, "error": str(e)}
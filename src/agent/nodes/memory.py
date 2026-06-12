import logging
from src.agent.state import AgentState
from src.memory.stm_manager import StmMemoryManager

logger = logging.getLogger(__name__)

_stm_mm = StmMemoryManager()


async def write_memory_node(state: AgentState) -> dict:
    """
    Persist the completed turn to STM (Redis). Always the final node.

    Uses state["answer"] as the assistant turn — which is populated by
    synthesize_node for rag/memory_only/rag_and_memory routes, and by
    clarify_node for the clarify route.

    If answer is empty (upstream failure), falls back to the error string
    so the turn is still recorded and the context window stays coherent.
    """
    answer = state.get("answer") or state.get("error") or "(no response)"
    meta = state.get("metadata", {})

    # Attach routing info to the turn metadata — useful when debugging
    # why the agent answered the way it did for a given turn.
    meta["route"] = state.get("route", "unknown")
    meta["route_reasoning"] = state.get("route_reasoning", "")

    try:
        await _stm_mm.write_stm(
            session_id=state["session_id"],
            user_text=state["question"],
            assistant_text=answer,
            meta=meta,
        )
        logger.info(f"write_memory_node: turn written for session={state['session_id']}")
    except Exception as e:
        logger.error(f"write_memory_node failed: {e}")
        return {"error": str(e)}

    return {}   # {} denotes that the node executed its task(writing stm) and no state to update
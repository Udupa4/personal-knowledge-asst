import logging
from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.memory.stm_manager import StmMemoryManager
from src.utils.agent_utils import get_message_text

logger = logging.getLogger(__name__)

_stm_mm = StmMemoryManager()

def _extract_answer(state: AgentState) -> str:
    answer = state.get("answer")
    if answer and isinstance(answer, str):
        return answer

    messages = state.get("messages", [])
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        text = get_message_text(msg)
        if text:
            return text

    return state.get("error") or "(no response)"


async def write_memory_node(state: AgentState) -> dict:
    """
    Persist the completed turn to STM. Always the final node.
    Handles both clarify_node output (state["answer"]) and
    agent_node output (state["messages"][-1].content).
    """
    answer = _extract_answer(state)
    meta = state.get("metadata", {})
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

    return {"answer": answer}
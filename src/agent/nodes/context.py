import asyncio
import logging
from src.agent.state import AgentState
from src.memory.stm_manager import StmMemoryManager
from src.memory.ltm_manager import LtmManager

logger = logging.getLogger(__name__)

stm_mm = StmMemoryManager()
ltm_mm = LtmManager()


async def load_context_node(state: AgentState) -> dict:
    """
    Load STM (Redis) and LTM (ChromaDB) in parallel.

    Runs unconditionally before the router so that the router has full
    context when deciding how to answer. Both reads are concurrent via
    asyncio.gather — total latency is max(stm_latency, ltm_latency),
    not the sum.

    Returns:
        stm_context: last 6 turns for this session
        ltm_context: top-3 relevant LTM summaries for this user+question
    """
    try:
        stm_task = stm_mm.read_stm(state["session_id"], k=6)

        # LtmManager.retrieve is sync (ChromaDB is not async-native).
        # asyncio.to_thread offloads it to a thread pool so it doesn't
        # block the event loop while gather is running.
        ltm_task = asyncio.to_thread(
            ltm_mm.retrieve,
            state["question"],
            state["user_id"],
            3
        )

        stm_context, ltm_context = await asyncio.gather(stm_task, ltm_task)

        logger.info(
            f"load_context_node: session={state['session_id']} "
            f"stm_turns={len(stm_context)} ltm_memories={len(ltm_context)}"
        )
        return {"stm_context": stm_context, "ltm_context": ltm_context}

    except Exception as e:
        logger.error(f"load_context_node failed: {e}")
        return {"stm_context": [], "ltm_context": [], "error": str(e)}

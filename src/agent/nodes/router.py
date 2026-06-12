import logging
from pydantic import BaseModel, Field
from src.agent.state import AgentState
from src.llm.llm import get_llm

logger = logging.getLogger(__name__)

# ── Structured output schema ───────────────────────────────────────────────
# with_structured_output forces the LLM to return this exact shape.
# If it can't, the call raises immediately — no silent misroutes.

class RouterDecision(BaseModel):
    route: str = Field(
        description=(
            "One of: 'rag', 'memory_only', 'rag_and_memory', 'clarify'. "
            "Choose 'rag' if the question is about factual topics in the knowledge base. "
            "Choose 'memory_only' if the question is about past conversations or personal context. "
            "Choose 'rag_and_memory' if both knowledge and personal context are needed. "
            "Choose 'clarify' if the question is too vague or ambiguous to answer without more info."
        )
    )
    reasoning: str = Field(
        description="One sentence explaining why you chose this route."
    )


# ── Router prompt ──────────────────────────────────────────────────────────
# Keep this short and deterministic. The router should classify, not answer.
# Temperature is 0.0 (set in get_llm default) for consistent routing.

ROUTER_SYSTEM_PROMPT = """You are a query router for a personal knowledge assistant.
Your job is to classify the user's question into exactly one routing category.
You are NOT answering the question — only deciding how it should be handled.

Routing rules:
- "rag"            → question asks about factual topics, documents, or knowledge base content
- "memory_only"    → question asks about past conversations, what was discussed, or personal context
- "rag_and_memory" → question needs both knowledge base content AND personal/conversational context
- "clarify"        → question is too vague, ambiguous, or incomplete to route confidently

You will be given:
- The user's question
- A summary of recent conversation turns (STM), if any
- A summary of long-term memory about this user (LTM), if any

Use the context to inform your routing decision but do not answer the question itself."""


def _build_router_input(state: AgentState) -> str:
    """Assemble the human-turn content for the router LLM call."""
    parts = [f"Question: {state['question']}"]

    if state.get("stm_context"):
        recent = state["stm_context"][:3]   # only last 3 turns to keep prompt small
        parts.append("\nRecent conversation (most recent first):")
        for turn in recent:
            parts.append(f"  User: {turn.get('user', '')}")
            parts.append(f"  Assistant: {turn.get('assistant', '')}")

    if state.get("ltm_context"):
        parts.append("\nLong-term memory about this user:")
        for mem in state["ltm_context"]:
            parts.append(f"  - {mem}")

    return "\n".join(parts)


async def router_node(state: AgentState) -> dict:
    """
    Classify the question into a routing strategy.

    Uses structured output to guarantee a valid route is returned.
    Writes 'route' and 'route_reasoning' into state — the conditional
    edge in graph.py reads 'route' to dispatch to the correct next node.
    """
    try:
        llm = get_llm()

        # Bind structured output — LLM must return a RouterDecision or raise
        structured_llm = llm.with_structured_output(RouterDecision)

        messages = [
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", _build_router_input(state)),
        ]

        decision: RouterDecision = await structured_llm.ainvoke(messages)

        logger.info(
            f"router_node: route={decision.route!r} "
            f"reasoning={decision.reasoning!r}"
        )

        return {
            "route": decision.route,
            "route_reasoning": decision.reasoning,
        }

    except Exception as e:
        logger.error(f"router_node failed: {e}. Defaulting to 'rag'.")
        # Safe fallback — rag is the most broadly useful route
        return {
            "route": "rag",
            "route_reasoning": f"Router failed ({e}), defaulted to rag.",
            "error": str(e),
        }

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
        description="Either 'agent' (attempt to answer) or 'clarify' (too vague)."
    )
    reasoning: str = Field(
        description="One sentence explaining why you chose this route."
    )


# ── Router prompt ──────────────────────────────────────────────────────────
# Keep this short and deterministic. The router should classify, not answer.
# Temperature is 0.0 (set in get_llm default) for consistent routing.

ROUTER_SYSTEM_PROMPT = """You are a query router for a personal knowledge assistant.
Classify the user's question into exactly one of two categories:

- "agent"   → the question can be answered (possibly with tool use)
- "clarify" → the question is too vague or incomplete to attempt an answer

You are NOT answering the question — only deciding whether to attempt it.
Output "clarify" only when the question is genuinely unanswerable without 
more information (e.g. single words, pronouns with no referent, pure noise).
When in doubt, output "agent" — the agent has tools to find what it needs.
"""


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

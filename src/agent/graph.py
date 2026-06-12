import logging
from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState
from src.agent.nodes.context import load_context_node
from src.agent.nodes.router import router_node
from src.agent.nodes.retrieve import retrieve_node
from src.agent.nodes.synthesize import synthesize_node
from src.agent.nodes.clarify import clarify_node
from src.agent.nodes.memory import write_memory_node

logger = logging.getLogger(__name__)

# Valid routes the router can return.
VALID_ROUTES = {"rag", "memory_only", "rag_and_memory", "clarify"}

def _route_dispatcher(state: AgentState) -> str:
    route = state.get("route", "rag")
    if route not in VALID_ROUTES:
        logger.warning(f"Unknown route '{route}', falling back to 'rag'.")
        return "rag"
    return route


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    builder.add_node("load_context", load_context_node)
    builder.add_node("router", router_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("write_memory", write_memory_node)

    # ── Edges ──────────────────────────────────────────────────────────────
    builder.add_edge(START, "load_context")     # Entry point
    builder.add_edge("load_context", "router")

    # Conditional edge through router node
    builder.add_conditional_edges(
        "router",
        _route_dispatcher,
        {
            "rag": "retrieve",
            "rag_and_memory": "retrieve",  # same retrieve node, both routes
            "memory_only": "synthesize",  # skip retrieval entirely
            "clarify": "clarify",
        },
    )

    # Both rag and rag_and_memory go through retrieve before synthesize.
    # memory_only skips retrieve and goes straight to synthesize.
    builder.add_edge("retrieve", "synthesize")
    builder.add_edge("synthesize", "write_memory")
    builder.add_edge("clarify", "write_memory")
    builder.add_edge("write_memory", END)

    return builder.compile()

graph = build_graph()

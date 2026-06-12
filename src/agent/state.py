from typing import TypedDict, List, Optional, Dict, Any


class AgentState(TypedDict):
    # ── Input fields (set once at graph entry, never mutated) ──────────────
    question: str
    session_id: str
    user_id: str
    top_k: int

    # ── Router decision ────────────────────────────────────────────────────
    # Written by router_node, consumed by the conditional edge dispatcher.
    # Possible values: "rag" | "memory_only" | "rag_and_memory" | "clarify"
    route: str
    route_reasoning: str    # LLM's explanation — surfaced in LangSmith traces

    # ── Context (written by load_context_node) ────────────────────────────
    stm_context: List[Dict[str, Any]]   # recent turns from Redis
    ltm_context: List[str]              # bullet-point summaries from ChromaDB

    # ── Retrieval (written by retrieve_node) ──────────────────────────────
    # Empty list when route = "memory_only"
    retrieved_docs: List[Dict[str, Any]]

    # ── Generation (written by synthesize_node or clarify_node) ──────────
    answer: str
    metadata: Dict[str, Any]            # token counts, model name, finish reason

    # ── Clarification ──────────────────────────────────────────────────────
    # Populated only when route = "clarify"
    clarification_question: Optional[str]

    # ── Error handling ─────────────────────────────────────────────────────
    # Any node can write here on failure. Downstream nodes check this
    # and degrade gracefully instead of crashing the graph.
    error: Optional[str]

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.agent.tools import UserContext
from src.agent.state import AgentState
from src.agent.nodes.context import load_context_node
from src.agent.nodes.router import router_node
from src.agent.nodes.agent import agent_node
from src.agent.nodes.clarify import clarify_node
from src.agent.nodes.memory import write_memory_node
from src.agent.tools import get_tools

logger = logging.getLogger(__name__)

# Valid routes the router can return.
VALID_ROUTES = {"agent", "clarify"}

def _route_dispatcher(state: AgentState) -> str:
    """
    Reads the router's decision and dispatches to the correct node.
    """
    route = state.get("route", "agent")
    # Router prompt will be updated to only output "agent" or "clarify"
    if route in {"rag", "rag_and_memory", "memory_only", "agent"}:
        return "agent"
    if route == "clarify":
        return "clarify"
    logger.warning(f"Unknown route '{route}', defaulting to 'agent'.")
    return "agent"

def _tools_condition_with_memory(state: AgentState) -> str:
    """
    Custom replacement for tools_condition that routes to write_memory
    instead of END when the agent produces its final answer.

    tools_condition logic:
      - last message has tool_calls → "tools"  (continue the loop)
      - last message has no tool_calls → "write_memory"  (done)
    """
    messages = state.get("messages", [])
    if not messages:
        return "write_memory"
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "write_memory"


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState, context_schema=UserContext)

    # Pass the callable directly into ToolNode instead of a static list
    tool_executor = ToolNode(get_tools())

    # ── Nodes ──────────────────────────────────────────────────────────────
    builder.add_node("load_context", load_context_node)
    builder.add_node("router", router_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_executor)
    builder.add_node("clarify", clarify_node)
    builder.add_node("write_memory", write_memory_node)

    # ── Edges ──────────────────────────────────────────────────────────────
    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "router")

    # Router dispatches to agent or clarify
    builder.add_conditional_edges(
        "router",
        _route_dispatcher,
        {
            "agent": "agent",
            "clarify": "clarify",
        },
    )

    # Tool call loop — agent either calls tools or produces final answer
    builder.add_conditional_edges(
        "agent",
        _tools_condition_with_memory,
        {
            "tools": "tools",
            "write_memory": "write_memory",
        },
    )

    # After tool execution, always return to agent for next pass
    builder.add_edge("tools", "agent")

    # Clarify bypasses the tool loop entirely
    builder.add_edge("clarify", "write_memory")
    builder.add_edge("write_memory", END)

    return builder.compile()

graph = build_graph()
# from PIL import Image
# from io import BytesIO
#
# png_graph = graph.get_graph().draw_mermaid_png()
# img = Image.open(BytesIO(png_graph))
# img.show()
# img.save("./src/agent/graph.png")

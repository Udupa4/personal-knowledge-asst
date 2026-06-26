import logging
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.tools import get_tools
from src.llm.llm import get_llm
from src.llm.prompt_template import template

logger = logging.getLogger(__name__)

def _build_initial_messages(state: AgentState) -> list:
    parts = []

    if state.get("ltm_context"):
        parts.append("## Long-term memory about this user (past sessions):")
        for mem in state["ltm_context"]:
            parts.append(f"  - {mem}")

    if state.get("stm_context"):
        parts.append("\n## Recent conversation history (for context only, "
                     "do not treat as knowledge source):")
        for turn in state["stm_context"]:
            parts.append(f"  User: {turn.get('user', '')}")
            parts.append(f"  Assistant: {turn.get('assistant', '')}")

    parts.append(f"\n## Current question (answer this, use tools if needed):")
    parts.append(state["question"])

    return [
        SystemMessage(content=template),
        HumanMessage(content="\n".join(parts)),
    ]

async def agent_node(state: AgentState) -> dict:
    """
    Core agent node — LLM with tools bound.

    First pass: builds the initial messages from state context and question.
    Subsequent passes (after tool execution): state["messages"] already
    contains the full history including tool results — pass them directly.

    The LLM responds with either:
      - tool_calls populated → tools_condition routes to tool_executor
      - tool_calls empty     → final answer, tools_condition routes to END
                               (actually to write_memory via our custom edge)
    """
    try:
        tools = get_tools()
        llm = get_llm().bind_tools(tools)

        # If messages is empty this is the first pass — build from state.
        # If messages exist we're in the tool loop — use accumulated history.
        if not state.get("messages"):
            initial_messages = _build_initial_messages(state)
            response = await llm.ainvoke(initial_messages)
            return {"messages": initial_messages + [response]}
        else:
            messages = state["messages"]
            response = await llm.ainvoke(messages)
            return {"messages": [response]}

    except Exception as e:
        logger.error(f"agent_node failed: {e}")
        from langchain_core.messages import AIMessage
        fallback = AIMessage(content="I encountered an error while processing your request.")
        return {"messages": [fallback], "error": str(e)}
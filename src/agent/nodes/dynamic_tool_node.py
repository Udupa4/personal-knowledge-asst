from src.agent.tools import get_tools
from src.agent.state import AgentState

async def dynamic_tool_node(state: AgentState) -> dict:
    """A custom, dynamic replacement for ToolNode."""
    # 1. Get the tools dynamically using the actual user_id in flight
    tools = get_tools(state["user_id"])
    tool_map = {tool.name: tool for tool in tools}

    # 2. Get the last AI message containing tool calls
    messages = state.get("messages", [])
    last_message = messages[-1]

    tool_outputs = []
    # 3. Execute the requested tools
    for tool_call in last_message.tool_calls:
        tool = tool_map.get(tool_call["name"])
        if tool:
            # Run the tool with its arguments
            output = await tool.ainvoke(tool_call["args"])

            # Create a properly formatted ToolMessage response
            from langchain_core.messages import ToolMessage
            tool_outputs.append(
                ToolMessage(
                    content=str(output),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]  # CRITICAL: Must match Gemini's requested ID!
                )
            )

    return {"messages": tool_outputs}
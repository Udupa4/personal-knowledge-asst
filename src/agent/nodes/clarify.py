import logging

from src.agent.state import AgentState
from src.llm.llm import get_llm

logger = logging.getLogger(__name__)

CLARIFY_SYSTEM_PROMPT = """You are a helpful assistant.
The user's question is too vague or ambiguous to answer well.
Generate exactly one concise clarifying question to help understand what they need.
Output only the clarifying question — no preamble, no explanation."""


async def clarify_node(state: AgentState) -> dict:
    try:
        llm = get_llm()
        response = await llm.ainvoke([
            ("system", CLARIFY_SYSTEM_PROMPT),
            ("human", state["question"])
        ])
        clarification_q = response.content.strip()

        logger.info(f"clarify_node: generated clarification={clarification_q!r}")
        return {
            "clarification_question": clarification_q,
            "answer": clarification_q,
            "metadata": {}
        }
    except Exception as e:
        logger.error(f"Error while executing clarify_node: {e}")
        fallback = "Could you please clarify your question?"
        return {
            "clarification_question": fallback,
            "answer": fallback,
            "metadata": {},
            "error": str(e),
        }
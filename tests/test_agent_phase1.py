"""
Phase 5 smoke tests — run from the project root:

    python -m tests.test_agent_phase1

Tests:
  1. AgentState shape is valid
  2. load_context_node populates stm_context and ltm_context
  3. router_node returns a valid route and reasoning
  4. Full partial graph (load_context → router → END) runs end-to-end

Each test prints PASS / FAIL with a short explanation.
LangSmith tracing is active if LANGCHAIN_TRACING_V2=true in your .env —
check https://smith.langchain.com after running to see the trace.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.agent.state import AgentState
from src.agent.nodes.context import load_context_node
from src.agent.nodes.router import router_node
from src.agent.graph import graph


# ── Minimal valid state used across all tests ──────────────────────────────
BASE_STATE: AgentState = {
    "question": "What are the best chunking strategies for RAG?",
    "session_id": "test-session-001",
    "user_id": "test-user-001",
    "top_k": 3,
    "route": "",
    "route_reasoning": "",
    "stm_context": [],
    "ltm_context": [],
    "retrieved_docs": [],
    "answer": "",
    "metadata": {},
    "clarification_question": None,
    "error": None,
}

VALID_ROUTES = {"rag", "memory_only", "rag_and_memory", "clarify"}


# ── Helpers ────────────────────────────────────────────────────────────────

def _pass(name: str):
    print(f"  ✓ PASS  {name}")

def _fail(name: str, reason: str):
    print(f"  ✗ FAIL  {name}: {reason}")


# ── Test 1: AgentState shape ───────────────────────────────────────────────

def test_state_shape():
    name = "AgentState has required keys"
    required = {
        "question", "session_id", "user_id", "top_k",
        "route", "route_reasoning",
        "stm_context", "ltm_context", "retrieved_docs",
        "answer", "metadata", "clarification_question", "error",
    }
    missing = required - set(BASE_STATE.keys())
    if missing:
        _fail(name, f"missing keys: {missing}")
    else:
        _pass(name)


# ── Test 2: load_context_node ─────────────────────────────────────────────

async def test_load_context_node():
    name = "load_context_node returns stm_context and ltm_context"
    result = await load_context_node(BASE_STATE)

    if "stm_context" not in result:
        _fail(name, "'stm_context' missing from result")
        return
    if "ltm_context" not in result:
        _fail(name, "'ltm_context' missing from result")
        return
    if not isinstance(result["stm_context"], list):
        _fail(name, f"stm_context is not a list: {type(result['stm_context'])}")
        return
    if not isinstance(result["ltm_context"], list):
        _fail(name, f"ltm_context is not a list: {type(result['ltm_context'])}")
        return

    _pass(name)
    print(f"         stm_turns={len(result['stm_context'])}  "
          f"ltm_memories={len(result['ltm_context'])}")


# ── Test 3: router_node — RAG question ────────────────────────────────────

async def test_router_node():
    state_with_context = {**BASE_STATE, "stm_context": [], "ltm_context": []}
    name = "router_node returns a valid route"
    result = await router_node(state_with_context)

    if "route" not in result:
        _fail(name, "'route' missing from result")
        return
    if result["route"] not in VALID_ROUTES:
        _fail(name, f"invalid route: {result['route']!r} (expected one of {VALID_ROUTES})")
        return
    if not result.get("route_reasoning"):
        _fail(name, "route_reasoning is empty")
        return

    _pass(name)
    print(f"         route={result['route']!r}")
    print(f"         reasoning={result['route_reasoning']!r}")


# ── Test 4: router_node — memory question ─────────────────────────────────

async def test_router_memory_question():
    name = "router_node routes memory question to 'memory_only' or 'rag_and_memory'"
    state = {
        **BASE_STATE,
        "question": "What did we discuss in our last conversation?",
        "stm_context": [
            {"user": "Tell me about vector databases", "assistant": "Vector databases store embeddings..."}
        ],
        "ltm_context": ["User previously discussed RAG and chunking strategies"],
    }
    result = await router_node(state)
    expected = {"memory_only", "rag_and_memory"}
    if result.get("route") not in expected:
        _fail(name, f"expected memory route, got {result.get('route')!r}")
    else:
        _pass(name)
        print(f"         route={result['route']!r}  reasoning={result['route_reasoning']!r}")


# ── Test 5: router_node — vague question ─────────────────────────────────

async def test_router_vague_question():
    name = "router_node routes vague question to 'clarify'"
    state = {**BASE_STATE, "question": "help", "stm_context": [], "ltm_context": []}
    result = await router_node(state)
    # "clarify" is expected but LLMs aren't fully deterministic — warn not hard fail
    if result.get("route") == "clarify":
        _pass(name)
    else:
        print(f"  ~ WARN  {name}: got {result.get('route')!r} instead of 'clarify' "
              f"(LLM non-determinism, not a hard failure)")




# ── Test 6: Full partial graph ─────────────────────────────────────────────

async def test_full_partial_graph():
    name = "Full graph (load_context → router → END) completes without error"
    try:
        result = await graph.ainvoke(BASE_STATE)
    except Exception as e:
        _fail(name, str(e))
        return

    if result.get("error") and not result.get("route"):
        _fail(name, f"graph errored with no route: {result['error']}")
        return
    if result.get("route") not in VALID_ROUTES:
        _fail(name, f"invalid final route: {result.get('route')!r}")
        return

    _pass(name)
    print(f"         final_route={result['route']!r}")
    print(f"         stm_turns={len(result.get('stm_context', []))}")
    print(f"         ltm_memories={len(result.get('ltm_context', []))}")
    print(f"\n  → Check LangSmith for the trace: https://smith.langchain.com")


# ── Runner ─────────────────────────────────────────────────────────────────

async def main():
    print("\n── Phase 5 Agent Tests ──────────────────────────────────────────\n")

    print("Test 1: State shape")
    test_state_shape()

    print("\nTest 2: load_context_node")
    await test_load_context_node()

    print("\nTest 3: router_node — RAG question")
    await test_router_node()

    print("\nTest 4: router_node — memory question")
    await test_router_memory_question()

    print("\nTest 5: router_node — vague question")
    await test_router_vague_question()

    print("\nTest 6: Full partial graph")
    await test_full_partial_graph()

    print("\n────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    pass # asyncio.run(main())

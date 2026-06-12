"""
    python -m tests.test_agent_phase2

Coverage:
  Unit tests  — each node in isolation with controlled state input
  Integration — full graph.ainvoke() for every route
  Edge cases  — error handling, empty context, fallback behavior

LangSmith traces will appear at https://smith.langchain.com for any test
that invokes the graph or an LLM-backed node. Note: The api key need to be present in the .env
"""

import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.agent.state import AgentState
from src.agent.nodes.context import load_context_node
from src.agent.nodes.router import router_node
from src.agent.nodes.retrieve import retrieve_node
from src.agent.nodes.synthesize import synthesize_node
from src.agent.nodes.clarify import clarify_node
from src.agent.nodes.memory import write_memory_node
from src.agent.graph import graph


# ── Test result helpers ────────────────────────────────────────────────────

PASS = 0
FAIL = 0

def _pass(name: str, detail: str = ""):
    global PASS
    PASS += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  ✓ PASS  {name}{suffix}")

def _fail(name: str, reason: str):
    global FAIL
    FAIL += 1
    print(f"  ✗ FAIL  {name}: {reason}")

def _warn(name: str, reason: str):
    print(f"  ~ WARN  {name}: {reason}")


# ── Shared fixtures ────────────────────────────────────────────────────────

VALID_ROUTES = {"rag", "memory_only", "rag_and_memory", "clarify"}

def _base_state(**overrides) -> AgentState:
    """
    Returns a minimal valid AgentState. Use keyword overrides to customize
    specific fields per test without repeating the full dict every time.
    """
    state: AgentState = {
        "question": "What are the best chunking strategies for RAG?",
        "session_id": f"test-{uuid.uuid4()}",   # unique per test — avoids STM cross-contamination
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
    state.update(overrides)
    return state


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — load_context_node
# ══════════════════════════════════════════════════════════════════════════

async def test_load_context_returns_lists():
    """Node always returns list types even for a brand-new session with no history."""
    name = "load_context: returns list types for empty session"
    result = await load_context_node(_base_state())
    if not isinstance(result.get("stm_context"), list):
        _fail(name, f"stm_context type={type(result.get('stm_context'))}")
        return
    if not isinstance(result.get("ltm_context"), list):
        _fail(name, f"ltm_context type={type(result.get('ltm_context'))}")
        return
    _pass(name, f"stm={len(result['stm_context'])} ltm={len(result['ltm_context'])}")


async def test_load_context_no_error_on_empty():
    """No error field should be set when session legitimately has no history."""
    name = "load_context: no error for brand-new session"
    result = await load_context_node(_base_state())
    if result.get("error"):
        _fail(name, f"unexpected error: {result['error']}")
    else:
        _pass(name)


async def test_load_context_reads_existing_stm():
    """
    After writing a turn directly to STM, load_context should return it.
    This verifies the node actually reads from Redis, not just returns [].
    """
    name = "load_context: reads existing STM turns from Redis"
    from src.memory.stm_manager import StmMemoryManager
    stm = StmMemoryManager()
    session_id = f"test-{uuid.uuid4()}"

    # Pre-populate one turn
    await stm.write_stm(session_id, "What is RAG?", "RAG stands for Retrieval-Augmented Generation.")

    state = _base_state(session_id=session_id)
    result = await load_context_node(state)

    # Cleanup
    await stm.clear_session(session_id)

    if len(result.get("stm_context", [])) != 1:
        _fail(name, f"expected 1 turn, got {len(result.get('stm_context', []))}")
        return
    turn = result["stm_context"][0]
    if turn.get("user") != "What is RAG?":
        _fail(name, f"turn content mismatch: {turn}")
        return
    _pass(name)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — router_node
# ══════════════════════════════════════════════════════════════════════════

async def test_router_returns_valid_route():
    """Any question must produce one of the four valid routes."""
    name = "router: returns a valid route for a factual question"
    result = await router_node(_base_state())
    route = result.get("route")
    if route not in VALID_ROUTES:
        _fail(name, f"invalid route: {route!r}")
        return
    if not result.get("route_reasoning"):
        _fail(name, "route_reasoning is empty")
        return
    _pass(name, f"route={route!r}")


async def test_router_rag_question():
    """
    A clear factual question about a technical topic should route to
    'rag' or 'rag_and_memory', not 'memory_only' or 'clarify'.
    """
    name = "router: factual question routes to 'rag' or 'rag_and_memory'"
    state = _base_state(question="What is the difference between ChromaDB and Pinecone?")
    result = await router_node(state)
    expected = {"rag", "rag_and_memory"}
    if result.get("route") not in expected:
        _warn(name, f"got {result.get('route')!r} — LLM non-determinism, verify manually")
    else:
        _pass(name, f"route={result['route']!r}")


async def test_router_memory_question_with_context():
    """
    A question explicitly about past conversations, combined with populated
    STM context, should route to 'memory_only' or 'rag_and_memory'.
    """
    name = "router: memory question with STM context routes correctly"
    state = _base_state(
        question="What did we discuss in our last conversation?",
        stm_context=[
            {"user": "Tell me about vector databases",
             "assistant": "Vector databases store high-dimensional embeddings..."}
        ],
        ltm_context=["User previously explored RAG and asked about chunking strategies"],
    )
    result = await router_node(state)
    expected = {"memory_only", "rag_and_memory"}
    if result.get("route") not in expected:
        _warn(name, f"got {result.get('route')!r} instead of a memory route")
    else:
        _pass(name, f"route={result['route']!r}")


async def test_router_vague_question():
    """Single-word or nonsensical input should trigger 'clarify'."""
    name = "router: vague question triggers 'clarify'"
    state = _base_state(question="help", stm_context=[], ltm_context=[])
    result = await router_node(state)
    if result.get("route") == "clarify":
        _pass(name)
    else:
        _warn(name, f"got {result.get('route')!r} — LLM non-determinism, not a hard failure")


async def test_router_always_has_reasoning():
    """route_reasoning must be non-empty for any input — it's logged in LangSmith."""
    name = "router: route_reasoning is always populated"
    questions = [
        "What is chunking?",
        "What did we talk about before?",
        "???",
    ]
    for q in questions:
        result = await router_node(_base_state(question=q))
        if not result.get("route_reasoning"):
            _fail(name, f"empty reasoning for question={q!r}")
            return
    _pass(name, f"checked {len(questions)} questions")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — retrieve_node
# ══════════════════════════════════════════════════════════════════════════

async def test_retrieve_returns_list():
    """retrieve_node must always return a list, never None."""
    name = "retrieve: returns a list"
    result = await retrieve_node(_base_state())
    if not isinstance(result.get("retrieved_docs"), list):
        _fail(name, f"retrieved_docs type={type(result.get('retrieved_docs'))}")
    else:
        _pass(name, f"docs={len(result['retrieved_docs'])}")


async def test_retrieve_relevant_question():
    """A question clearly in the knowledge base should return at least one doc."""
    name = "retrieve: relevant question returns docs"
    state = _base_state(question="What are chunking strategies in RAG?", top_k=3)
    result = await retrieve_node(state)
    docs = result.get("retrieved_docs", [])
    if len(docs) == 0:
        _warn(name, "no docs returned — check Chroma store is populated (run /ingest first)")
        return
    _pass(name, f"docs={len(docs)}")


async def test_retrieve_doc_shape():
    """Each returned doc must have the expected keys."""
    name = "retrieve: returned docs have correct shape"
    state = _base_state(question="What is a vector database?", top_k=2)
    result = await retrieve_node(state)
    docs = result.get("retrieved_docs", [])
    if not docs:
        _warn(name, "no docs to inspect — skipping shape check")
        return
    required_keys = {"title", "filename", "snippet"}
    for i, doc in enumerate(docs):
        missing = required_keys - set(doc.keys())
        if missing:
            _fail(name, f"doc[{i}] missing keys: {missing}")
            return
    _pass(name, f"checked {len(docs)} docs")


async def test_retrieve_top_k_respected():
    """retrieve_node must never return more docs than top_k."""
    name = "retrieve: respects top_k limit"
    state = _base_state(question="What is RAG?", top_k=2)
    result = await retrieve_node(state)
    docs = result.get("retrieved_docs", [])
    if len(docs) > 2:
        _fail(name, f"expected ≤2 docs, got {len(docs)}")
    else:
        _pass(name, f"docs={len(docs)} top_k=2")


async def test_retrieve_no_error_field_on_success():
    """On a successful retrieval, 'error' must not be set."""
    name = "retrieve: no error field on successful call"
    result = await retrieve_node(_base_state())
    if result.get("error"):
        _fail(name, f"unexpected error: {result['error']}")
    else:
        _pass(name)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — synthesize_node
# ══════════════════════════════════════════════════════════════════════════

async def test_synthesize_with_docs():
    """synthesize_node with retrieved docs should produce a non-empty answer."""
    name = "synthesize: produces answer when docs are available"
    state = _base_state(
        question="What is chunking in RAG?",
        retrieved_docs=[{
            "title": "RAG Fundamentals",
            "filename": "rag_fundamentals.txt",
            "snippet": "Chunking splits documents into smaller pieces for embedding and retrieval.",
            "path": "./data/docs/rag_fundamentals.txt",
        }],
    )
    result = await synthesize_node(state)
    answer = result.get("answer", "")
    if not answer or answer.startswith("Encountered an error"):
        _fail(name, f"bad answer: {answer!r}")
        return
    _pass(name, f"answer_len={len(answer)}")


async def test_synthesize_without_docs():
    """
    synthesize_node with no retrieved_docs (memory_only route) must still
    produce a coherent answer from STM/LTM context alone.
    """
    name = "synthesize: works without retrieved docs (memory_only route)"
    state = _base_state(
        question="What did we talk about last time?",
        retrieved_docs=[],
        stm_context=[
            {"user": "Tell me about async Python",
             "assistant": "Async Python uses asyncio for concurrent I/O..."}
        ],
        ltm_context=["User previously studied async patterns and RAG fundamentals"],
    )
    result = await synthesize_node(state)
    answer = result.get("answer", "")
    if not answer or answer.startswith("Encountered an error"):
        _fail(name, f"bad answer: {answer!r}")
        return
    _pass(name, f"answer_len={len(answer)}")


async def test_synthesize_returns_metadata():
    """metadata dict must be present and not empty after a successful call."""
    name = "synthesize: returns non-empty metadata"
    result = await synthesize_node(_base_state(question="What is RAG?"))
    metadata = result.get("metadata", {})
    if not isinstance(metadata, dict):
        _fail(name, f"metadata type={type(metadata)}")
        return
    # Metadata contents vary by provider — just check it's populated
    if not metadata:
        _warn(name, "metadata is empty — may be provider-specific behaviour")
    else:
        _pass(name, f"keys={list(metadata.keys())}")


async def test_synthesize_empty_state():
    """
    synthesize_node with completely empty context must not crash.
    It should degrade gracefully and return some answer.
    """
    name = "synthesize: handles fully empty context without crashing"
    state = _base_state(
        question="What is a transformer?",
        retrieved_docs=[],
        stm_context=[],
        ltm_context=[],
    )
    result = await synthesize_node(state)
    if "answer" not in result:
        _fail(name, "'answer' key missing from result")
    else:
        _pass(name, f"answer_len={len(result['answer'])}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — clarify_node
# ══════════════════════════════════════════════════════════════════════════

async def test_clarify_produces_question():
    """clarify_node must generate a non-empty string that looks like a question."""
    name = "clarify: generates a non-empty clarifying question"
    result = await clarify_node(_base_state(question="help"))
    clarification = result.get("clarification_question", "")
    if not clarification:
        _fail(name, "clarification_question is empty")
        return
    _pass(name, f"clarification={clarification!r}")


async def test_clarify_answer_matches_clarification():
    """
    clarify_node must write the same text to both 'answer' and
    'clarification_question' — write_memory_node reads 'answer'.
    """
    name = "clarify: 'answer' matches 'clarification_question'"
    result = await clarify_node(_base_state(question="I need help with something"))
    if result.get("answer") != result.get("clarification_question"):
        _fail(name,
              f"answer={result.get('answer')!r} != "
              f"clarification_question={result.get('clarification_question')!r}")
    else:
        _pass(name)


async def test_clarify_vague_input():
    """Single-word input should still produce a meaningful clarifying question."""
    name = "clarify: handles single-word vague input"
    result = await clarify_node(_base_state(question="explain"))
    clarification = result.get("clarification_question", "")
    if not clarification:
        _fail(name, "empty clarification for vague input")
    else:
        _pass(name, f"clarification={clarification!r}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — write_memory_node
# ══════════════════════════════════════════════════════════════════════════

async def test_write_memory_persists_turn():
    """
    After write_memory_node runs, read_stm should return the written turn.
    This is the most important test for this node — verify the Redis write
    actually happened, don't just check the return value of the node.
    """
    name = "write_memory: turn is actually persisted to Redis"
    from src.memory.stm_manager import StmMemoryManager
    stm = StmMemoryManager()
    session_id = f"test-{uuid.uuid4()}"

    state = _base_state(
        session_id=session_id,
        question="What is LTM?",
        answer="LTM stands for Long-Term Memory.",
        route="rag",
        route_reasoning="Factual question about memory systems.",
        metadata={"model_name": "test-model"},
    )

    await write_memory_node(state)

    # Read back from Redis and verify
    turns = await stm.read_stm(session_id, k=1)
    await stm.clear_session(session_id)     # cleanup

    if len(turns) != 1:
        _fail(name, f"expected 1 turn in Redis, got {len(turns)}")
        return
    turn = turns[0]
    if turn.get("user") != "What is LTM?":
        _fail(name, f"user field mismatch: {turn.get('user')!r}")
        return
    if turn.get("assistant") != "LTM stands for Long-Term Memory.":
        _fail(name, f"assistant field mismatch: {turn.get('assistant')!r}")
        return
    _pass(name)


async def test_write_memory_attaches_route_to_meta():
    """Route and route_reasoning must be stored inside the turn's meta field."""
    name = "write_memory: route info attached to turn metadata"
    from src.memory.stm_manager import StmMemoryManager
    stm = StmMemoryManager()
    session_id = f"test-{uuid.uuid4()}"

    state = _base_state(
        session_id=session_id,
        question="What is chunking?",
        answer="Chunking splits documents into smaller pieces.",
        route="rag",
        route_reasoning="Factual question.",
        metadata={},
    )

    await write_memory_node(state)
    turns = await stm.read_stm(session_id, k=1)
    await stm.clear_session(session_id)

    if not turns:
        _fail(name, "no turn written to Redis")
        return
    meta = turns[0].get("meta", {})
    if meta.get("route") != "rag":
        _fail(name, f"route not in meta: {meta}")
        return
    if not meta.get("route_reasoning"):
        _fail(name, f"route_reasoning not in meta: {meta}")
        return
    _pass(name)


async def test_write_memory_fallback_on_empty_answer():
    """
    If 'answer' is empty (upstream failure), write_memory_node must still
    write something to Redis — it falls back to error or '(no response)'.
    """
    name = "write_memory: falls back gracefully when answer is empty"
    from src.memory.stm_manager import StmMemoryManager
    stm = StmMemoryManager()
    session_id = f"test-{uuid.uuid4()}"

    state = _base_state(
        session_id=session_id,
        question="What is RAG?",
        answer="",          # simulates upstream failure
        error="LLM timed out",
        route="rag",
        route_reasoning="",
        metadata={},
    )

    await write_memory_node(state)
    turns = await stm.read_stm(session_id, k=1)
    await stm.clear_session(session_id)

    if not turns:
        _fail(name, "no turn written even on empty answer")
        return
    assistant_text = turns[0].get("assistant", "")
    if not assistant_text:
        _fail(name, "assistant field is empty in Redis — fallback didn't work")
        return
    _pass(name, f"fallback_text={assistant_text!r}")


async def test_write_memory_returns_empty_dict():
    """write_memory_node is a side effect node — it must return {}."""
    name = "write_memory: returns empty dict (pure side-effect node)"
    session_id = f"test-{uuid.uuid4()}"
    from src.memory.stm_manager import StmMemoryManager
    stm = StmMemoryManager()

    state = _base_state(
        session_id=session_id,
        answer="Some answer.",
        route="rag",
        metadata={},
    )
    result = await write_memory_node(state)
    await stm.clear_session(session_id)

    if result != {}:
        _fail(name, f"expected {{}}, got {result}")
    else:
        _pass(name)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — Full graph integration
# ══════════════════════════════════════════════════════════════════════════

async def _run_graph(question: str, **overrides) -> AgentState:
    """Helper — runs the full graph and returns the final state."""
    state = _base_state(question=question, **overrides)
    return await graph.ainvoke(state)


async def test_graph_rag_route():
    """
    Full graph run for a knowledge-base question.
    Expected: route=rag or rag_and_memory, retrieved_docs populated, answer present.
    """
    name = "graph[rag]: end-to-end RAG question"
    result = await _run_graph("What are the chunking strategies for RAG documents?")

    if result.get("route") not in {"rag", "rag_and_memory"}:
        _warn(name, f"route={result.get('route')!r} — LLM may have chosen differently")

    if not result.get("answer"):
        _fail(name, "answer is empty")
        return
    _pass(name,
          f"route={result.get('route')!r} "
          f"docs={len(result.get('retrieved_docs', []))} "
          f"answer_len={len(result.get('answer', ''))}")


async def test_graph_memory_only_route():
    """
    Full graph run for a memory question with pre-populated STM.
    Expected: route=memory_only or rag_and_memory, retrieved_docs empty or small.
    """
    name = "graph[memory_only]: conversation history question"
    from src.memory.stm_manager import StmMemoryManager
    stm = StmMemoryManager()
    session_id = f"test-{uuid.uuid4()}"

    # Pre-populate STM so the question has real context to work with
    await stm.write_stm(session_id,
                        "What is a vector database?",
                        "A vector database stores high-dimensional embeddings.")
    await stm.write_stm(session_id,
                        "Which one should I use for prototyping?",
                        "ChromaDB is ideal for local prototyping.")

    result = await _run_graph(
        "Can you summarise what we've discussed so far?",
        session_id=session_id,
    )
    await stm.clear_session(session_id)

    if not result.get("answer"):
        _fail(name, "answer is empty")
        return
    if result.get("route") not in {"memory_only", "rag_and_memory"}:
        _warn(name, f"route={result.get('route')!r} — expected a memory route")

    _pass(name,
          f"route={result.get('route')!r} "
          f"answer_len={len(result.get('answer', ''))}")


async def test_graph_clarify_route():
    """
    Full graph run for a vague one-word question.
    Expected: route=clarify, clarification_question populated.
    """
    name = "graph[clarify]: vague question triggers clarification"
    result = await _run_graph("help")

    if result.get("route") != "clarify":
        _warn(name, f"route={result.get('route')!r} — expected 'clarify'")
        return

    if not result.get("clarification_question"):
        _fail(name, "clarification_question is empty even though route=clarify")
        return
    if not result.get("answer"):
        _fail(name, "answer is empty — clarify_node should write answer too")
        return

    _pass(name,
          f"clarification={result.get('clarification_question')!r}")


async def test_graph_turn_written_to_stm():
    """
    After any full graph run, the turn must be readable from STM.
    This verifies write_memory_node executed at the end of every route.
    """
    name = "graph: completed turn is written to STM after every route"
    from src.memory.stm_manager import StmMemoryManager
    stm = StmMemoryManager()
    session_id = f"test-{uuid.uuid4()}"

    await _run_graph(
        "What is the difference between STM and LTM?",
        session_id=session_id,
    )

    turns = await stm.read_stm(session_id, k=1)
    await stm.clear_session(session_id)

    if not turns:
        _fail(name, "no turn found in STM after graph run")
        return
    if not turns[0].get("assistant"):
        _fail(name, "assistant field empty in written turn")
        return
    _pass(name, f"turn written: user={turns[0].get('user', '')[:40]!r}")


async def test_graph_state_fields_populated():
    """
    After a full graph run, all expected output fields must be set.
    Nothing should be left at its initial empty value.
    """
    name = "graph: all output state fields populated after run"
    result = await _run_graph("What are the best practices for RAG chunking?")

    failures = []
    if not result.get("route"):
        failures.append("route is empty")
    if not result.get("route_reasoning"):
        failures.append("route_reasoning is empty")
    if not result.get("answer"):
        failures.append("answer is empty")
    if result.get("stm_context") is None:
        failures.append("stm_context is None")
    if result.get("ltm_context") is None:
        failures.append("ltm_context is None")
    if result.get("retrieved_docs") is None:
        failures.append("retrieved_docs is None")

    if failures:
        _fail(name, ", ".join(failures))
    else:
        _pass(name, f"route={result.get('route')!r}")


async def test_graph_error_does_not_crash():
    """
    Even if the question is malformed or edge-case, the graph must complete
    without raising an exception — errors are captured in state['error'].
    """
    name = "graph: completes without raising for edge-case inputs"
    edge_cases = [
        "",             # empty string
        "?" * 10,       # nonsense punctuation
        "a" * 500,      # very long single token
    ]
    for q in edge_cases:
        try:
            result = await _run_graph(q)
            # graph should complete — answer or error should be set
            if not result.get("answer") and not result.get("error"):
                _fail(name, f"both answer and error are empty for input={q[:30]!r}")
                return
        except Exception as e:
            _fail(name, f"graph raised exception for input={q[:30]!r}: {e}")
            return
    _pass(name, f"tested {len(edge_cases)} edge cases")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — Multi-turn STM coherence
# ══════════════════════════════════════════════════════════════════════════

async def test_multiturn_context_accumulates():
    """
    Run three questions on the same session in sequence. After each run,
    STM should contain one more turn. The third question can reference the
    first two and the router should consider them.
    """
    name = "multi-turn: STM accumulates across consecutive graph runs"
    from src.memory.stm_manager import StmMemoryManager
    stm = StmMemoryManager()
    session_id = f"test-{uuid.uuid4()}"

    questions = [
        "What is a vector database?",
        "Which vector database is best for prototyping?",
        "What did I just ask about?",
    ]

    for i, q in enumerate(questions):
        await _run_graph(q, session_id=session_id)
        turns = await stm.read_stm(session_id, k=10)
        if len(turns) != i + 1:
            await stm.clear_session(session_id)
            _fail(name, f"after question {i+1}, expected {i+1} turns, got {len(turns)}")
            return

    await stm.clear_session(session_id)
    _pass(name, f"session accumulated {len(questions)} turns correctly")


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════

async def main():
    print("\n── Phase 5 Full Test Suite ──────────────────────────────────────\n")

    sections = [
        # ("1. load_context_node", [
        #     test_load_context_returns_lists,
        #     test_load_context_no_error_on_empty,
        #     test_load_context_reads_existing_stm,
        # ]),
        # ("2. router_node", [
        #     test_router_returns_valid_route,
        #     test_router_rag_question,
        #     test_router_memory_question_with_context,
        #     test_router_vague_question,
        #     test_router_always_has_reasoning,
        # ]),
        # ("3. retrieve_node", [
        #     test_retrieve_returns_list,
        #     test_retrieve_relevant_question,
        #     test_retrieve_doc_shape,
        #     test_retrieve_top_k_respected,
        #     test_retrieve_no_error_field_on_success,
        # ]),
        # ("4. synthesize_node", [
        #     test_synthesize_with_docs,
        #     test_synthesize_without_docs,
        #     test_synthesize_returns_metadata,
        #     test_synthesize_empty_state,
        # ]),
        # ("5. clarify_node", [
        #     test_clarify_produces_question,
        #     test_clarify_answer_matches_clarification,
        #     test_clarify_vague_input,
        # ]),
        # ("6. write_memory_node", [
        #     test_write_memory_persists_turn,
        #     test_write_memory_attaches_route_to_meta,
        #     test_write_memory_fallback_on_empty_answer,
        #     test_write_memory_returns_empty_dict,
        # ]),
        ("7. Full graph integration", [
            test_graph_rag_route,
            test_graph_memory_only_route,
            test_graph_clarify_route,
            test_graph_turn_written_to_stm,
            test_graph_state_fields_populated,
            test_graph_error_does_not_crash,
        ]),
        ("8. Multi-turn coherence", [
            test_multiturn_context_accumulates,
        ]),
    ]

    for section_name, tests in sections:
        print(f"\nSection {section_name}")
        for test_fn in tests:
            try:
                await test_fn()
            except Exception as e:
                _fail(test_fn.__name__, f"unhandled exception: {e}")

    print(f"\n────────────────────────────────────────────────────────────────")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("  All tests passed. Check LangSmith for traces.")
    else:
        print("  Fix failures before proceeding to Phase 6.")
    print()


if __name__ == "__main__":
    asyncio.run(main())

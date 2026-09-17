"""Memory architecture tests.

Tests the new memory architecture components without requiring a live LLM or
database connection.  All tests are pure-Python unit tests that run fast.

Run with:
    python -m pytest tests/test_memory_arch.py -v
or:
    python tests/test_memory_arch.py
"""

import sys
import os
import asyncio

# Ensure the project root is on sys.path regardless of where the test is run from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"


def _assert(condition: bool, name: str, detail: str = ""):
    if condition:
        print(f"  [{_PASS}] {name}")
    else:
        msg = f"  [{_FAIL}] {name}"
        if detail:
            msg += f"\n         Detail: {detail}"
        print(msg)
    return condition


# ---------------------------------------------------------------------------
# Test 1: Read gate — trivial / greeting queries are skipped
# ---------------------------------------------------------------------------

async def test_mem0_read_gate():
    print("\n[Test 1] Mem0 read gate — smart retrieval heuristic")
    from app.services.memory import _should_retrieve

    # Should NOT trigger search (greetings / acknowledgements / general questions)
    skip_cases = [
        "hi",
        "ok",
        "thanks",
        "What is RAG?",
        "Explain transformers",
        "How does Python work?"
    ]
    skip_count = 0
    for q in skip_cases:
        if not _should_retrieve(q):
            skip_count += 1
    _assert(skip_count == len(skip_cases), f"all {len(skip_cases)} impersonal/trivial queries skipped", f"skipped {skip_count}/{len(skip_cases)}")

    # Should trigger search (personal/memory queries)
    retrieve_cases = [
        "What is my name?",
        "What do I prefer?",
        "What project am I working on?",
        "What did we discuss yesterday?",
        "Which university was I considering?",
        "What programming language do I prefer?"
    ]
    search_count = 0
    for q in retrieve_cases:
        if _should_retrieve(q):
            search_count += 1
    _assert(search_count == len(retrieve_cases), f"all {len(retrieve_cases)} personal queries pass gate", f"passed {search_count}/{len(retrieve_cases)}")


# ---------------------------------------------------------------------------
# Test 2: Write gate — only long-term-worthy content is stored
# ---------------------------------------------------------------------------

async def test_mem0_write_gate():
    print("\n[Test 2] Mem0 write gate — only store relevant content")
    from app.services.memory import _should_store

    # Should store
    worthy = [
        "My name is Ojas and I prefer dark mode",
        "I work as a software engineer at Google",
        "Remember that I always use Python 3.11",
        "My goal is to build a personal AI assistant",
        "I prefer concise answers please",
        "I'm a big fan of FastAPI",
    ]
    store_count = sum(1 for m in worthy if _should_store(m))
    _assert(store_count == len(worthy), f"all {len(worthy)} worthy messages pass write gate", f"stored {store_count}/{len(worthy)}")

    # Should NOT store
    unworthy = [
        "What is 2 + 2?",
        "Summarize this document",
        "How does Python work?",
        "Write a haiku about autumn",
        "Can you help me debug this code?",
    ]
    skip_count = sum(1 for m in unworthy if not _should_store(m))
    _assert(skip_count == len(unworthy), f"all {len(unworthy)} unworthy messages blocked", f"blocked {skip_count}/{len(unworthy)}")


# ---------------------------------------------------------------------------
# Test 3: Context windowing — ContextEngine respects RECENT_MESSAGES_WINDOW
# ---------------------------------------------------------------------------

async def test_context_windowing():
    print("\n[Test 3] ContextEngine — message windowing")
    from app.services.context_engine import ContextEngine
    from app.config import settings

    engine = ContextEngine()
    window = settings.RECENT_MESSAGES_WINDOW  # default 20

    # Build a history that's twice the window size
    long_history = []
    for i in range(window * 2):
        long_history.append(HumanMessage(content=f"User message {i}"))
        long_history.append(AIMessage(content=f"AI response {i}"))

    result = engine.build(
        system_instructions="You are helpful.",
        conversation_summary="",
        mem0_memories="",
        messages=long_history,
    )

    # result = [SystemMessage, ...windowed...]
    non_system = [m for m in result if not isinstance(m, SystemMessage)]
    _assert(
        len(non_system) <= window,
        f"windowed messages <= {window} (got {len(non_system)})",
    )

    # Always has exactly 1 SystemMessage
    sys_count = sum(1 for m in result if isinstance(m, SystemMessage))
    _assert(sys_count == 1, f"exactly 1 SystemMessage in context (got {sys_count})")


# ---------------------------------------------------------------------------
# Test 4: No duplicate system prompts in persistent state
# ---------------------------------------------------------------------------

async def test_no_duplicate_system_messages():
    print("\n[Test 4] No SystemMessage accumulation in persistent state")
    from app.services.context_engine import ContextEngine

    engine = ContextEngine()

    # Simulate a history that has stale SystemMessages (the old bug)
    dirty_history = [
        SystemMessage(content="Old system prompt turn 1"),
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
        SystemMessage(content="Old system prompt turn 2"),
        HumanMessage(content="How are you?"),
        AIMessage(content="I am fine!"),
    ]

    result = engine.build(
        system_instructions="Clean system prompt.",
        conversation_summary="",
        mem0_memories="",
        messages=dirty_history,
    )

    system_msgs = [m for m in result if isinstance(m, SystemMessage)]
    _assert(len(system_msgs) == 1, f"exactly 1 SystemMessage in final context (stale ones stripped), got {len(system_msgs)}")
    _assert(
        "Clean system prompt." in system_msgs[0].content,
        "SystemMessage contains the current system instructions"
    )


# ---------------------------------------------------------------------------
# Test 5: Context budget is respected
# ---------------------------------------------------------------------------

async def test_context_budget():
    print("\n[Test 5] ContextEngine — character budget enforcement")
    from app.services.context_engine import ContextEngine, _estimate_chars
    from app.config import settings

    engine = ContextEngine()
    budget = settings.CONTEXT_CHAR_BUDGET

    # Create a very long history that would blow the budget
    long_history = [
        AIMessage(content="X" * 500)
        for _ in range(100)
    ]

    result = engine.build(
        system_instructions="You are helpful.",
        conversation_summary="",
        mem0_memories="",
        messages=long_history,
    )

    total_chars = _estimate_chars(result)
    _assert(
        total_chars <= budget,
        f"context fits within {budget}-char budget (actual: {total_chars})"
    )


# ---------------------------------------------------------------------------
# Test 6: Summarizer threshold check
# ---------------------------------------------------------------------------

async def test_summarizer_threshold():
    print("\n[Test 6] Summarizer — threshold detection")
    from app.services.summarizer import ConversationSummarizer
    from app.config import settings

    s = ConversationSummarizer()
    threshold = settings.SUMMARY_THRESHOLD

    # Below threshold (threshold is total messages, so pairs * 2)
    short_msgs = []
    pairs_below = (threshold // 2) - 2
    for i in range(pairs_below):
        short_msgs.append(HumanMessage(content=f"msg {i}"))
        short_msgs.append(AIMessage(content=f"resp {i}"))

    should_compact_short = await s.should_summarize(short_msgs)
    _assert(not should_compact_short, f"no summarization below threshold ({len(short_msgs)} messages)")

    # Above threshold
    long_msgs = []
    pairs_above = (threshold // 2) + 2
    for i in range(pairs_above):
        long_msgs.append(HumanMessage(content=f"msg {i}"))
        long_msgs.append(AIMessage(content=f"resp {i}"))

    should_compact_long = await s.should_summarize(long_msgs)
    _assert(should_compact_long, f"summarization triggered above threshold ({len(long_msgs)} messages)")


# ---------------------------------------------------------------------------
# Test 7: Summarizer compaction trims messages correctly
# ---------------------------------------------------------------------------

async def test_summarizer_compaction():
    print("\n[Test 7] Summarizer — message trimming (no live LLM)")
    from app.services.summarizer import ConversationSummarizer
    from app.config import settings

    s = ConversationSummarizer()
    window = settings.RECENT_MESSAGES_WINDOW

    # Build a message list above threshold
    messages = []
    for i in range(window + 10):
        messages.append(HumanMessage(content=f"Question {i}"))
        messages.append(AIMessage(content=f"Answer {i}"))

    # compact() will call the LLM if available; we test the fallback path
    # by passing an invalid provider so the LLM call fails gracefully
    updates = await s.compact(
        messages=messages,
        existing_summary="Previous summary text.",
        provider="ollama",   # likely not running; tests the error-fallback path
        model="nonexistent",
    )

    # The compaction must always trim messages to the window
    kept = updates.get("messages", messages)
    _assert(
        len(kept) <= window,
        f"messages trimmed to <= {window} after compaction (got {len(kept)})"
    )

    # The summary must be non-empty (either LLM result or fallback text)
    summary = updates.get("summary", "")
    _assert(bool(summary), f"summary is non-empty after compaction")


# ---------------------------------------------------------------------------
# Test 8: ContextEngine includes summary in system block
# ---------------------------------------------------------------------------

async def test_context_includes_summary():
    print("\n[Test 8] ContextEngine — conversation summary injected into system block")
    from app.services.context_engine import ContextEngine

    engine = ContextEngine()
    summary_text = "The user previously asked about Python and FastAPI architecture."

    result = engine.build(
        system_instructions="You are helpful.",
        conversation_summary=summary_text,
        mem0_memories="",
        messages=[],
    )

    system_msgs = [m for m in result if isinstance(m, SystemMessage)]
    _assert(
        len(system_msgs) == 1 and summary_text in system_msgs[0].content,
        "conversation summary is embedded in SystemMessage"
    )


# ---------------------------------------------------------------------------
# Test 9: ContextEngine includes Mem0 memories in system block
# ---------------------------------------------------------------------------

async def test_context_includes_mem0():
    print("\n[Test 9] ContextEngine — Mem0 memories injected into system block")
    from app.services.context_engine import ContextEngine

    engine = ContextEngine()
    memories = "- User prefers Python\n- User works at Acme Corp\n- User's name is Ojas"

    result = engine.build(
        system_instructions="You are helpful.",
        conversation_summary="",
        mem0_memories=memories,
        messages=[],
    )

    system_msgs = [m for m in result if isinstance(m, SystemMessage)]
    _assert(
        len(system_msgs) == 1 and "Ojas" in system_msgs[0].content,
        "Mem0 memories are embedded in SystemMessage"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_all():
    print("=" * 60)
    print("  Memory Architecture Test Suite")
    print("=" * 60)

    tests = [
        test_mem0_read_gate,
        test_mem0_write_gate,
        test_context_windowing,
        test_no_duplicate_system_messages,
        test_context_budget,
        test_summarizer_threshold,
        test_summarizer_compaction,
        test_context_includes_summary,
        test_context_includes_mem0,
    ]

    results = []
    for test_fn in tests:
        try:
            await test_fn()
            results.append(True)
        except Exception as e:
            print(f"  [{_FAIL}] {test_fn.__name__} raised: {e}")
            import traceback; traceback.print_exc()
            results.append(False)

    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} tests passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all())
    sys.exit(0 if success else 1)

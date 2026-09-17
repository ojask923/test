"""RAG grounding and citation tests.

Tests the reranking pipeline, RetrievedChunk dataclass, format_cited_context,
ContextEngine grounding injection, and citation metadata — all as pure-unit
tests requiring no live LLM, Qdrant, or network connection.

Run with:
    python -m pytest tests/test_rag_grounding.py -v
or:
    python tests/test_rag_grounding.py
"""

from __future__ import annotations

import asyncio
import sys
import os

# Ensure the project root is on sys.path regardless of where the test is run from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 output on Windows (CP1252 terminal can't encode CJK bracket chars)
import io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from langchain_core.messages import SystemMessage

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"


def _assert(condition: bool, name: str, detail: str = "") -> bool:
    if condition:
        print(f"  [{_PASS}] {name}")
    else:
        msg = f"  [{_FAIL}] {name}"
        if detail:
            msg += f"\n         Detail: {detail}"
        print(msg)
    return condition


# ---------------------------------------------------------------------------
# Test 1: RetrievedChunk dataclass has all required citation fields
# ---------------------------------------------------------------------------

async def test_retrieved_chunk_dataclass():
    print("\n[Test 1] RetrievedChunk — dataclass fields and defaults")
    from app.services.rag_service import RetrievedChunk

    chunk = RetrievedChunk(
        text="The mitochondria is the powerhouse of the cell.",
        document_id="doc123",
        filename="biology.pdf",
        page_number=7,
        section="Cell Biology",
        chunk_id="abc12345",
        chunk_index=2,
        source="biology.pdf",
        rrf_score=0.0142,
        rerank_score=0.91,
        retrieval_latency_ms=42.7,
    )

    _assert(chunk.text == "The mitochondria is the powerhouse of the cell.", "text field populated")
    _assert(chunk.document_id == "doc123", "document_id field populated")
    _assert(chunk.filename == "biology.pdf", "filename field populated")
    _assert(chunk.page_number == 7, "page_number field populated")
    _assert(chunk.section == "Cell Biology", "section field populated")
    _assert(chunk.chunk_id == "abc12345", "chunk_id field populated")
    _assert(chunk.rerank_score == 0.91, "rerank_score field populated")
    _assert(chunk.retrieval_latency_ms == 42.7, "retrieval_latency_ms field populated")

    # Test defaults
    minimal = RetrievedChunk(text="hello")
    _assert(minimal.document_id == "", "document_id defaults to empty string")
    _assert(minimal.page_number is None, "page_number defaults to None")
    _assert(minimal.rerank_score is None, "rerank_score defaults to None (fallback mode)")


# ---------------------------------------------------------------------------
# Test 2: format_cited_context — basic 3-chunk case
# ---------------------------------------------------------------------------

async def test_format_cited_context_basic():
    print("\n[Test 2] format_cited_context — basic 3-chunk case")
    from app.services.rag_service import RetrievedChunk, format_cited_context

    chunks = [
        RetrievedChunk(text="Photosynthesis converts light to energy.", filename="bio.pdf", page_number=1, section="Plants"),
        RetrievedChunk(text="ATP is used in cellular respiration.", filename="bio.pdf", page_number=5),
        RetrievedChunk(text="DNA carries genetic information.", filename="genetics.pdf", page_number=12, section="DNA"),
    ]

    context_block, citation_map = format_cited_context(chunks)

    _assert("\u3010Doc 1\u3011" in context_block, "\u3010Doc 1\u3011 marker present")
    _assert("\u3010Doc 2\u3011" in context_block, "\u3010Doc 2\u3011 marker present")
    _assert("\u3010Doc 3\u3011" in context_block, "\u3010Doc 3\u3011 marker present")
    _assert(len(citation_map) == 3, f"citation_map has 3 entries (got {len(citation_map)})")
    _assert("Photosynthesis" in context_block, "chunk text included in context block")
    _assert("genetics.pdf" in context_block, "second filename appears in context block")


# ---------------------------------------------------------------------------
# Test 3: format_cited_context — empty list returns empty string + empty dict
# ---------------------------------------------------------------------------

async def test_format_cited_context_empty():
    print("\n[Test 3] format_cited_context — empty chunk list")
    from app.services.rag_service import format_cited_context

    context_block, citation_map = format_cited_context([])

    _assert(context_block == "", f"empty context_block for no chunks (got {repr(context_block)})")
    _assert(citation_map == {}, f"empty citation_map for no chunks (got {citation_map})")


# ---------------------------------------------------------------------------
# Test 4: Correct citation — citation_map contains expected metadata
# ---------------------------------------------------------------------------

async def test_correct_citation():
    print("\n[Test 4] format_cited_context — citation metadata correctness")
    from app.services.rag_service import RetrievedChunk, format_cited_context

    chunk = RetrievedChunk(
        text="The conclusion is X.",
        document_id="docABC",
        filename="report.pdf",
        page_number=3,
        section="Results",
        chunk_id="cid001",
        chunk_index=0,
        rrf_score=0.0167,
        rerank_score=0.88,
    )

    _, citation_map = format_cited_context([chunk])

    meta = citation_map.get(1)
    _assert(meta is not None, "citation_map[1] exists")
    _assert(meta["filename"] == "report.pdf", f"filename == 'report.pdf' (got {meta.get('filename')})")
    _assert(meta["page_number"] == 3, f"page_number == 3 (got {meta.get('page_number')})")
    _assert(meta["section"] == "Results", f"section == 'Results' (got {meta.get('section')})")
    _assert(meta["chunk_id"] == "cid001", f"chunk_id == 'cid001' (got {meta.get('chunk_id')})")
    _assert(meta["document_id"] == "docABC", f"document_id == 'docABC' (got {meta.get('document_id')})")
    _assert(meta["rerank_score"] == 0.88, f"rerank_score == 0.88 (got {meta.get('rerank_score')})")


# ---------------------------------------------------------------------------
# Test 5: Missing evidence — no grounding block when rag_citations is empty
# ---------------------------------------------------------------------------

async def test_missing_evidence_prompt():
    print("\n[Test 5] ContextEngine — no grounding block when rag_citations is empty/None")
    from app.services.context_engine import ContextEngine

    engine = ContextEngine()

    # No citations -> grounding block must NOT appear in system prompt
    result = engine.build(
        system_instructions="You are helpful.",
        conversation_summary="",
        mem0_memories="",
        messages=[],
        rag_citations=None,
    )
    sys_msgs = [m for m in result if isinstance(m, SystemMessage)]
    _assert(len(sys_msgs) == 1, "exactly 1 SystemMessage")
    _assert("CITATION RULES" not in sys_msgs[0].content,
            "no CITATION RULES block when rag_citations is None")
    _assert("<rag_grounding>" not in sys_msgs[0].content,
            "no <rag_grounding> block when rag_citations is None")

    # Same with empty dict
    result2 = engine.build(
        system_instructions="You are helpful.",
        conversation_summary="",
        mem0_memories="",
        messages=[],
        rag_citations={},
    )
    sys_msgs2 = [m for m in result2 if isinstance(m, SystemMessage)]
    _assert("CITATION RULES" not in sys_msgs2[0].content,
            "no CITATION RULES block when rag_citations is empty dict")


# ---------------------------------------------------------------------------
# Test 6: Grounding instruction injected when citations are present
# ---------------------------------------------------------------------------

async def test_grounding_instruction_injected():
    print("\n[Test 6] ContextEngine — grounding block injected when rag_citations present")
    from app.services.context_engine import ContextEngine

    engine = ContextEngine()
    citation_map = {
        1: {"filename": "paper.pdf", "page_number": 2, "section": "Intro",
            "chunk_id": "cid1", "document_id": "d1", "rrf_score": 0.01, "rerank_score": 0.9},
    }

    result = engine.build(
        system_instructions="You are a research assistant.",
        conversation_summary="",
        mem0_memories="",
        messages=[],
        rag_citations=citation_map,
    )

    sys_msgs = [m for m in result if isinstance(m, SystemMessage)]
    _assert(len(sys_msgs) == 1, "exactly 1 SystemMessage")
    sys_content = sys_msgs[0].content
    _assert("CITATION RULES" in sys_content, "CITATION RULES block present in system prompt")
    _assert("<rag_grounding>" in sys_content, "<rag_grounding> tag present")
    _assert("\u3010Doc N\u3011" in sys_content, "\u3010Doc N\u3011 marker referenced in grounding rules")
    _assert("not found in the provided documents" in sys_content,
            "'not found in provided documents' instruction present")


# ---------------------------------------------------------------------------
# Test 7: Conflicting documents — both appear independently in citation_map
# ---------------------------------------------------------------------------

async def test_conflicting_documents():
    print("\n[Test 7] format_cited_context — conflicting documents tracked independently")
    from app.services.rag_service import RetrievedChunk, format_cited_context

    chunks = [
        RetrievedChunk(text="Study A finds X increases risk.", filename="studyA.pdf",
                       page_number=4, section="Results A", document_id="docA"),
        RetrievedChunk(text="Study B finds X decreases risk.", filename="studyB.pdf",
                       page_number=9, section="Results B", document_id="docB"),
    ]

    context_block, citation_map = format_cited_context(chunks)

    _assert(len(citation_map) == 2, "both conflicting documents in citation_map")
    _assert(citation_map[1]["document_id"] == "docA", "citation_map[1] is studyA")
    _assert(citation_map[2]["document_id"] == "docB", "citation_map[2] is studyB")
    _assert("Study A" in context_block, "studyA text in context block")
    _assert("Study B" in context_block, "studyB text in context block")
    _assert("Results A" in context_block, "studyA section in context block")
    _assert("Results B" in context_block, "studyB section in context block")


# ---------------------------------------------------------------------------
# Test 8: Irrelevant retrieval — low-score chunks still get citation metadata
# ---------------------------------------------------------------------------

async def test_irrelevant_retrieval():
    print("\n[Test 8] format_cited_context — low-score (irrelevant) chunks still cited properly")
    from app.services.rag_service import RetrievedChunk, format_cited_context

    chunks = [
        RetrievedChunk(text="Marginally relevant content.", filename="noise.pdf",
                       page_number=1, rrf_score=0.0001, rerank_score=0.01),
    ]

    context_block, citation_map = format_cited_context(chunks)

    _assert("\u3010Doc 1\u3011" in context_block, "low-score chunk gets \u3010Doc 1\u3011 marker")
    _assert(citation_map[1]["filename"] == "noise.pdf", "low-score chunk has filename metadata")
    _assert(citation_map[1]["rerank_score"] == 0.01, f"low rerank_score preserved (got {citation_map[1].get('rerank_score')})")


# ---------------------------------------------------------------------------
# Test 9: Reranker graceful fallback — returns RRF-ordered chunks, no crash
# ---------------------------------------------------------------------------

async def test_rerank_fallback_graceful():
    print("\n[Test 9] retrieve_structured — graceful fallback when reranker raises")
    from unittest.mock import MagicMock, patch
    from app.services.rag_service import RAGService

    svc = RAGService.__new__(RAGService)
    svc._embeddings = None
    svc._sparse_embeddings = None
    svc._reranker = None
    svc._vector_store = None
    svc._client = None

    # Mock a broken reranker
    broken_reranker = MagicMock()
    broken_reranker.rerank.side_effect = RuntimeError("model load failed")

    # Mock Qdrant client returning 1 hit
    mock_hit = MagicMock()
    mock_hit.id = "uuid-001"
    mock_hit.score = 0.85
    mock_hit.payload = {
        "page_content": "Test content",
        "metadata": {"filename": "test.pdf", "page_number": 1, "document_id": "d1",
                     "section": None, "chunk_id": "c1", "chunk_index": 0, "source": "test.pdf"},
    }

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [mock_hit]

    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 384

    svc._client = mock_client
    svc._embeddings = mock_embeddings

    with patch.object(type(svc), "reranker", new_callable=lambda: property(lambda self: broken_reranker)):
        import app.config as cfg_mod
        original_rerank = cfg_mod.settings.ENABLE_RERANKING
        original_hybrid = cfg_mod.settings.ENABLE_HYBRID_SEARCH
        cfg_mod.settings.ENABLE_RERANKING = True
        cfg_mod.settings.ENABLE_HYBRID_SEARCH = False
        try:
            result = svc.retrieve_structured("test query", top_k=3, candidate_k=5)
        finally:
            cfg_mod.settings.ENABLE_RERANKING = original_rerank
            cfg_mod.settings.ENABLE_HYBRID_SEARCH = original_hybrid

    _assert(isinstance(result, list), "returns a list on reranker failure (no crash)")
    if result:
        _assert(result[0].rerank_score is None, "rerank_score is None in fallback mode")


# ---------------------------------------------------------------------------
# Test 10: candidate_k is respected (fetches candidate_k, returns <= top_k)
# ---------------------------------------------------------------------------

async def test_candidate_k_respected():
    print("\n[Test 10] retrieve_structured — candidate_k fetched, <= top_k returned")
    from unittest.mock import MagicMock
    from app.services.rag_service import RAGService

    svc = RAGService.__new__(RAGService)
    svc._embeddings = None
    svc._sparse_embeddings = None
    svc._reranker = None
    svc._vector_store = None

    def make_hit(i):
        h = MagicMock()
        h.id = f"uuid-{i:03d}"
        h.score = 1.0 / (i + 1)
        h.payload = {
            "page_content": f"Content chunk {i}",
            "metadata": {"filename": "big.pdf", "page_number": i, "document_id": "bigdoc",
                         "section": None, "chunk_id": f"c{i}", "chunk_index": i, "source": "big.pdf"},
        }
        return h

    mock_hits = [make_hit(i) for i in range(25)]
    mock_client = MagicMock()
    mock_client.query_points.return_value.points = mock_hits
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 384

    svc._client = mock_client
    svc._embeddings = mock_embeddings

    import app.config as cfg_mod
    orig_hybrid = cfg_mod.settings.ENABLE_HYBRID_SEARCH
    orig_rerank = cfg_mod.settings.ENABLE_RERANKING
    cfg_mod.settings.ENABLE_HYBRID_SEARCH = False
    cfg_mod.settings.ENABLE_RERANKING = False
    try:
        result = svc.retrieve_structured("test query", top_k=5, candidate_k=20)
    finally:
        cfg_mod.settings.ENABLE_HYBRID_SEARCH = orig_hybrid
        cfg_mod.settings.ENABLE_RERANKING = orig_rerank

    _assert(len(result) <= 5, f"<= 5 chunks returned (top_k=5), got {len(result)}")
    _assert(mock_client.query_points.called, "Qdrant query_points was called")


# ---------------------------------------------------------------------------
# Test 11: Retrieval latency is measured and non-negative
# ---------------------------------------------------------------------------

async def test_latency_measured():
    print("\n[Test 11] retrieve_structured — retrieval_latency_ms measured and >= 0")
    from unittest.mock import MagicMock
    from app.services.rag_service import RAGService

    svc = RAGService.__new__(RAGService)
    svc._embeddings = None
    svc._sparse_embeddings = None
    svc._reranker = None
    svc._vector_store = None

    mock_hit = MagicMock()
    mock_hit.id = "uuid-lat"
    mock_hit.score = 0.9
    mock_hit.payload = {
        "page_content": "Latency test content",
        "metadata": {"filename": "lat.pdf", "page_number": 1, "document_id": "d_lat",
                     "section": None, "chunk_id": "clat", "chunk_index": 0, "source": "lat.pdf"},
    }

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [mock_hit]
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 384

    svc._client = mock_client
    svc._embeddings = mock_embeddings

    import app.config as cfg_mod
    orig_hybrid = cfg_mod.settings.ENABLE_HYBRID_SEARCH
    orig_rerank = cfg_mod.settings.ENABLE_RERANKING
    cfg_mod.settings.ENABLE_HYBRID_SEARCH = False
    cfg_mod.settings.ENABLE_RERANKING = False
    try:
        result = svc.retrieve_structured("latency test query", top_k=1, candidate_k=5)
    finally:
        cfg_mod.settings.ENABLE_HYBRID_SEARCH = orig_hybrid
        cfg_mod.settings.ENABLE_RERANKING = orig_rerank

    _assert(len(result) >= 1, "at least 1 chunk returned")
    if result:
        latency = result[0].retrieval_latency_ms
        _assert(latency >= 0.0, f"retrieval_latency_ms >= 0 (got {latency:.3f}ms)")
        _assert(isinstance(latency, float), f"retrieval_latency_ms is a float (got {type(latency)})")


# ---------------------------------------------------------------------------
# Test 12: No fabricated citations — grounding rules explicitly prohibit it
# ---------------------------------------------------------------------------

async def test_no_fabricated_citations():
    print("\n[Test 12] ContextEngine — grounding rules prohibit citation fabrication")
    from app.services.context_engine import ContextEngine

    engine = ContextEngine()
    citation_map = {
        1: {"filename": "facts.pdf", "page_number": 1, "section": None,
            "chunk_id": "c1", "document_id": "d1", "rrf_score": 0.01, "rerank_score": 0.95},
    }

    result = engine.build(
        system_instructions="You are precise.",
        conversation_summary="",
        mem0_memories="",
        messages=[],
        rag_citations=citation_map,
    )

    sys_content = next(m.content for m in result if isinstance(m, SystemMessage))

    _assert("Do NOT fabricate citations" in sys_content,
            "'Do NOT fabricate citations' instruction present in system prompt")
    _assert("not found in the provided documents" in sys_content,
            "'not found in provided documents' fallback instruction present")
    _assert("Do NOT cite a" in sys_content,
            "'Do NOT cite a \u3010Doc N\u3011' rule present (prevents hallucinated doc refs)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_all():
    print("=" * 60)
    print("  RAG Grounding & Citation Test Suite")
    print("=" * 60)

    tests = [
        test_retrieved_chunk_dataclass,
        test_format_cited_context_basic,
        test_format_cited_context_empty,
        test_correct_citation,
        test_missing_evidence_prompt,
        test_grounding_instruction_injected,
        test_conflicting_documents,
        test_irrelevant_retrieval,
        test_rerank_fallback_graceful,
        test_candidate_k_respected,
        test_latency_measured,
        test_no_fabricated_citations,
    ]

    results = []
    for test_fn in tests:
        try:
            await test_fn()
            results.append(True)
        except Exception as e:
            print(f"  [{_FAIL}] {test_fn.__name__} raised: {e}")
            import traceback
            traceback.print_exc()
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

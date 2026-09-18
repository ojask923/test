"""LangGraph agent workflow with multi-provider LLM support and PostgresSaver checkpointing.

Memory architecture
-------------------
* AgentState extends MessagesState with a ``summary`` field that stores the
  rolling compact summary of older conversation turns.
* Context is assembled by ContextEngine at model-call time — never injected
  through config or accumulated in persistent state as SystemMessages.
* A ``summarize`` node runs after each ``agent`` node invocation when the
  message count exceeds SUMMARY_THRESHOLD; it compacts old messages and
  updates ``AgentState.summary`` via the normal LangGraph state-update path.
* Mem0 search is called inside the graph node (gated by memory_service's
  read gate) — not outside in get_response/stream_response.
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional, Annotated

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from app.config import settings
from app.agent.tools import get_available_tools
from app.services.memory import memory_service
from app.services.database import db_service
from app.services.context_engine import context_engine
from app.services.summarizer import summarizer
from app.services.llm_factory import get_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB connection helper
# ---------------------------------------------------------------------------

def _build_pg_conn_string(db_url: str) -> str:
    """Convert a SQLAlchemy DATABASE_URL to a plain psycopg3 connection string."""
    conn = db_url
    for suffix in ("+psycopg2", "+psycopg", "+asyncpg", "+pg8000"):
        conn = conn.replace(f"postgresql{suffix}://", "postgresql://")
    conn = conn.replace("postgres://", "postgresql://")
    return conn


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """LangGraph state carrying message history, a rolling summary, and RAG citations.

    The ``summary`` field stores compact text produced by the Summarizer
    when the conversation grows beyond SUMMARY_THRESHOLD messages.
    The ``rag_citations`` field is populated by the retrieve_documents tool
    and contains the citation map from the last RAG retrieval.
    Both fields are persisted by PostgresSaver across restarts.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str
    rag_citations: dict  # citation_map from the last retrieve_structured() call


# ---------------------------------------------------------------------------
# LLM factory is now in app/services/llm_factory.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ChatAgent
# ---------------------------------------------------------------------------

class ChatAgent:
    """Manages LangGraph compilation and async-safe PostgresSaver checkpointing."""

    def __init__(self):
        self.checkpointer = None
        self._checkpointer_ready = False
        self._checkpointer_error: Optional[str] = None
        self._pool = None
        self.tools = get_available_tools()
        self._compiled_graphs: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Checkpointer lifecycle
    # ------------------------------------------------------------------

    async def _ensure_checkpointer(self) -> None:
        """Initialise PostgresSaver (or fall back to MemorySaver) exactly once."""
        if self._checkpointer_ready:
            return

        db_url = settings.DATABASE_URL
        is_postgres = db_url.startswith(("postgresql://", "postgresql+", "postgres://"))

        if is_postgres:
            try:
                import psycopg
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                from psycopg_pool import AsyncConnectionPool

                conn_string = _build_pg_conn_string(db_url)

                async with await psycopg.AsyncConnection.connect(
                    conn_string, autocommit=True
                ) as setup_conn:
                    await AsyncPostgresSaver(setup_conn).setup()

                self._pool = AsyncConnectionPool(
                    conninfo=conn_string,
                    min_size=1,
                    max_size=10,
                    open=False,
                )
                await self._pool.open(wait=True, timeout=10)
                self.checkpointer = AsyncPostgresSaver(self._pool)
                self._checkpointer_error = None
                logger.info("[Checkpointer] PostgresSaver ready — checkpoints persist across restarts.")
                print("[INFO] LangGraph checkpointer: PostgresSaver (persistent across restarts).")
            except Exception as exc:
                import traceback
                self._checkpointer_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "[Checkpointer] PostgresSaver init failed — falling back to MemorySaver.\n%s",
                    traceback.format_exc(),
                )
                print("[WARNING] PostgresSaver FAILED — using MemorySaver fallback.")
                print(f"[WARNING] Error: {type(exc).__name__}: {exc}")
                self.checkpointer = MemorySaver()
        else:
            print("[INFO] LangGraph checkpointer: MemorySaver (SQLite mode — no cross-restart persistence).")
            self.checkpointer = MemorySaver()

        self._checkpointer_ready = True
        self._compiled_graphs.clear()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ):
        """Construct a compiled StateGraph for the selected provider and model."""
        assert self.checkpointer is not None, "call await _ensure_checkpointer() before build_graph()"

        provider = provider or settings.DEFAULT_PROVIDER
        cache_key = f"{provider}_{model}_{temperature}"

        if cache_key in self._compiled_graphs:
            return self._compiled_graphs[cache_key]

        tools = self.tools if settings.ENABLE_TOOLS else []
        llm = get_llm(provider, model, temperature, tools=tools)
        tool_node = ToolNode(self.tools)

        # ---- agent node ---------------------------------------------------
        async def call_model(state: AgentState, config: RunnableConfig):
            """Build context via ContextEngine and call the LLM."""
            configurable = config.get("configurable", {})
            system_instructions = configurable.get(
                "system_prompt", "You are a helpful, smart AI assistant."
            )
            user_id = configurable.get("user_id", "")

            messages = state["messages"]

            # Extract the last HumanMessage's text as the current query for Mem0 search
            current_query = ""
            for i in range(len(messages) - 1, -1, -1):
                m = messages[i]
                if isinstance(m, HumanMessage):
                    content = m.content
                    if isinstance(content, str):
                        current_query = content
                    elif isinstance(content, list):
                        current_query = " ".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    break

            # Retrieve Mem0 long-term memories (read gate is inside memory_service)
            mem0_memories = ""
            if user_id and current_query:
                try:
                    mem0_memories = await memory_service.search(
                        user_id=user_id, query=current_query
                    )
                except Exception as exc:
                    logger.warning("[call_model] Mem0 search failed: %s", exc)

            # Build the final context via ContextEngine
            # Pass any RAG citations accumulated in state (from retrieve_documents tool calls)
            rag_citations = state.get("rag_citations", {})
            final_messages = context_engine.build(
                system_instructions=system_instructions,
                conversation_summary=state.get("summary", ""),
                mem0_memories=mem0_memories,
                messages=messages,
                rag_citations=rag_citations,
            )

            response = await llm.ainvoke(final_messages)
            return {"messages": [response]}

        # ---- summarize node -----------------------------------------------
        async def maybe_summarize(state: AgentState, config: RunnableConfig):
            """Compact old messages into a rolling summary when threshold is hit."""
            messages = state["messages"]
            should_compact = await summarizer.should_summarize(messages)
            if not should_compact:
                return {}

            configurable = config.get("configurable", {})
            updates = await summarizer.compact(
                messages=messages,
                existing_summary=state.get("summary", ""),
                provider=configurable.get("provider", provider),
                model=configurable.get("model", model),
            )
            logger.info(
                "[Graph] Summarization complete. Summary: %d chars. Messages kept: %d.",
                len(updates.get("summary", "")),
                len(updates.get("messages", messages)),
            )
            return updates

        # ---- routing ------------------------------------------------------
        def route_after_agent(state: AgentState):
            """Route to tools if the last message has tool calls, else summarize."""
            last = state["messages"][-1] if state["messages"] else None
            if last and hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "summarize"

        # ---- wire the graph -----------------------------------------------
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_node("summarize", maybe_summarize)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            route_after_agent,
            {"tools": "tools", "summarize": "summarize"},
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("summarize", END)

        compiled = workflow.compile(checkpointer=self.checkpointer)
        self._compiled_graphs[cache_key] = compiled
        return compiled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_response(
        self,
        message: str,
        session_id: str = "default",
        user_id: str = "default_user",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Invoke the graph and return the final reply."""
        await self._ensure_checkpointer()

        provider = provider or settings.DEFAULT_PROVIDER
        graph = self.build_graph(provider, model, temperature)

        config = {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id,
                "provider": provider,
                "model": model,
                "system_prompt": system_prompt or "You are a helpful, smart AI assistant.",
            }
        }

        # Only send the new HumanMessage; LangGraph loads prior state from checkpointer
        input_state: Dict[str, Any] = {
            "messages": [HumanMessage(content=message)],
            "summary": "",
            "rag_citations": {},
        }
        result = await graph.ainvoke(input_state, config=config)

        all_messages = result.get("messages", [])
        last_ai_msg = next(
            (m for m in reversed(all_messages) if isinstance(m, AIMessage) and m.content),
            None,
        )
        content = last_ai_msg.content if last_ai_msg else "No response generated."

        # Conditionally store to Mem0 in background (write gate is inside add())
        asyncio.create_task(
            memory_service.add(
                user_id=user_id,
                messages=[
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": content},
                ],
            )
        )

        # Extract citations from the final state
        raw_citations = result.get("rag_citations", {})
        citations = [
            {"index": idx, **meta}
            for idx, meta in raw_citations.items()
        ] if raw_citations else []

        return {
            "content": content,
            "session_id": session_id,
            "provider": provider,
            "model": model,
            "memories_used": False,
            "citations": citations,
        }

    async def stream_response(
        self,
        message: str,
        session_id: str = "default",
        user_id: str = "default_user",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response tokens and tool call notifications."""
        await self._ensure_checkpointer()

        provider_clean = (provider or settings.DEFAULT_PROVIDER).lower()
        graph = self.build_graph(provider_clean, model, temperature)

        config = {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id,
                "provider": provider_clean,
                "model": model,
                "system_prompt": system_prompt or "You are a helpful, smart AI assistant.",
            }
        }

        input_state: Dict[str, Any] = {
            "messages": [HumanMessage(content=message)],
            "summary": "",
            "rag_citations": {},
        }
        accumulated_text = ""

        async for event in graph.astream_events(input_state, config=config, version="v2"):
            kind = event.get("event")

            if kind == "on_chat_model_stream":
                tags = event.get("tags", [])
                if "query_rewriter" in tags:
                    continue
                
                chunk = event["data"].get("chunk")
                if chunk and chunk.content:
                    if isinstance(chunk.content, str):
                        accumulated_text += chunk.content
                        yield {"type": "token", "content": chunk.content}
                    elif isinstance(chunk.content, list):
                        for part in chunk.content:
                            if isinstance(part, dict) and "text" in part:
                                accumulated_text += part["text"]
                                yield {"type": "token", "content": part["text"]}

            elif kind == "on_tool_start":
                tool_name = event.get("name", "tool")
                tool_args = event.get("data", {}).get("input", {})
                yield {"type": "tool_start", "name": tool_name, "args": tool_args}

            elif kind == "on_tool_end":
                tool_name = event.get("name", "tool")
                tool_output = str(event.get("data", {}).get("output", ""))
                yield {"type": "tool_end", "name": tool_name, "result": tool_output}

        # Conditionally store to Mem0 in background
        if accumulated_text:
            asyncio.create_task(
                memory_service.add(
                    user_id=user_id,
                    messages=[
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": accumulated_text},
                    ],
                )
            )

        yield {"type": "done"}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve stored chat history for a session from the relational DB."""
        db_msgs = db_service.get_session_messages(session_id)
        return [
            {"role": m.role, "content": m.content}
            for m in db_msgs
            if m.role in ("user", "assistant") and m.content
        ]

    def clear_history(self, session_id: str) -> None:
        """Clear LangGraph checkpoint rows for a session.

        For PostgresSaver: deletes rows from the langgraph checkpoint tables.
        For MemorySaver:   pops from the in-process dict.
        Silently no-ops if the checkpointer has not yet been initialised.
        """
        if not self._checkpointer_ready or self.checkpointer is None:
            return

        if isinstance(self.checkpointer, MemorySaver):
            try:
                if hasattr(self.checkpointer, "storage"):
                    self.checkpointer.storage.pop(session_id, None)
            except Exception:
                pass
            return

        # PostgresSaver — delete via a short-lived sync psycopg3 connection
        try:
            import psycopg

            conn_string = _build_pg_conn_string(settings.DATABASE_URL)
            with psycopg.connect(conn_string) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (session_id,))
                    cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (session_id,))
                    cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (session_id,))
                conn.commit()
        except Exception as exc:
            logger.warning(
                "[clear_history] Could not delete Postgres checkpoints for %s: %s",
                session_id,
                exc,
            )


agent = ChatAgent()

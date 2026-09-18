"""ContextEngine: single layer that assembles the final LLM context each turn.

Combines:
  1. System instructions (static + dynamic persona)
  2. Conversation summary (compressed older history, stored in AgentState)
  3. Relevant Mem0 memories (long-term user facts, already retrieved and gated upstream)
  4. Windowed recent messages (last N from MessagesState)
  5. Current user query

Applies a configurable token budget so context never grows unbounded.
Uses tiktoken for accurate token counting against provider limits.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
import tiktoken

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Initialize tokenizer once for the module
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _count_tokens(messages: List[BaseMessage]) -> int:
    """Accurate token-count estimate for a list of messages."""
    total = 0
    for m in messages:
        content = m.content
        text_to_encode = ""
        if isinstance(content, str):
            text_to_encode = content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text_to_encode += str(part.get("text", ""))
        
        if text_to_encode:
            total += len(_tokenizer.encode(text_to_encode))
    return total


def _trim_to_budget(
    messages: List[BaseMessage],
    budget_tokens: int,
    keep_first: int = 1,
) -> List[BaseMessage]:
    """Drop the oldest *middle* messages until the total fits within budget_tokens.

    The first *keep_first* messages (e.g. SystemMessage) and the very last
    message (current query) are always preserved.
    """
    if _count_tokens(messages) <= budget_tokens:
        return messages

    protected_head = messages[:keep_first]
    protected_tail = messages[-1:]
    middle = list(messages[keep_first:-1])

    while middle and _count_tokens(protected_head + middle + protected_tail) > budget_tokens:
        middle.pop(0)  # drop the oldest middle message first

    trimmed = protected_head + middle + protected_tail
    logger.debug(
        "[ContextEngine] Context trimmed: %d -> %d messages to fit %d-token budget",
        len(messages),
        len(trimmed),
        budget_tokens,
    )
    return trimmed


# ---------------------------------------------------------------------------
# ContextEngine
# ---------------------------------------------------------------------------


class ContextEngine:
    """Assembles the final message list sent to the LLM each turn."""

    def build(
        self,
        *,
        system_instructions: str,
        conversation_summary: str,
        mem0_memories: str,
        messages: Sequence[BaseMessage],
        rag_citations: Optional[Dict[int, dict]] = None,
    ) -> List[BaseMessage]:
        """Return a fully assembled List[BaseMessage] ready for llm.ainvoke().

        Parameters
        ----------
        system_instructions:
            Static system prompt / persona text.
        conversation_summary:
            Rolling summary of older conversation turns. Empty string if none.
        mem0_memories:
            Pre-retrieved, pre-gated Mem0 long-term memories as formatted text.
        messages:
            Sequence of current conversation messages from MessagesState (including tool calls).
            Any stale SystemMessages are stripped out here.
        rag_citations:
            Optional citation map produced by :func:`format_cited_context`.
            When non-empty and ``settings.RAG_GROUNDING_ENABLED`` is True,
            a citation rules block is injected into the system prompt.
        """
        # --- 1. Build the composite system block ----------------------------
        system_parts: List[str] = [
            system_instructions.strip(),
            "\nIMPORTANT INSTRUCTIONS:\n- Do NOT output raw internal citation tags like `【retrieve_documents†source=1】`. Integrate the information naturally into your answer instead."
        ]

        if conversation_summary:
            system_parts.append(
                "<conversation_summary>\n"
                "The following is a summary of the older part of this conversation "
                "(older messages have been compacted to save space):\n"
                + conversation_summary.strip()
                + "\n</conversation_summary>"
            )

        if mem0_memories:
            system_parts.append(
                "<long_term_memory>\n"
                "The following are facts remembered about this user from past conversations:\n"
                + mem0_memories.strip()
                + "\n</long_term_memory>"
            )

        # --- 1b. Inject RAG citation grounding rules when documents are present ---
        grounding_enabled = getattr(settings, "RAG_GROUNDING_ENABLED", True)
        if grounding_enabled and rag_citations:
            system_parts.append(
                "<rag_grounding>\n"
                "The retrieved document chunks below were injected into this conversation via "
                "the retrieve_documents tool. Each chunk is identified by a \u3010Doc N\u3011 marker.\n\n"
                "CITATION RULES (follow strictly):\n"
                "1. When your answer uses information from a retrieved document, reference it "
                "inline as \u3010Doc N\u3011 (e.g. \u300caccording to \u3010Doc 1\u3011, the conclusion was...\u300d).\n"
                "2. If the retrieved documents do NOT contain sufficient evidence to answer the "
                "question, you MUST explicitly say: 'This information was not found in the "
                "provided documents.' Do NOT invent an answer.\n"
                "3. Do NOT fabricate citations. Do NOT cite a \u3010Doc N\u3011 that was not retrieved.\n"
                "4. Clearly distinguish between:\n"
                "   (a) information sourced from retrieved documents \u2192 cite with \u3010Doc N\u3011\n"
                "   (b) information from your general training knowledge \u2192 note it as general knowledge\n"
                "   (c) information from web search results \u2192 note the source URL where available\n"
                "</rag_grounding>"
            )

        system_content = "\n\n".join(system_parts)
        system_msg = SystemMessage(content=system_content)

        # --- 2. Strip any stale SystemMessages from history -----------------
        filtered_messages: List[BaseMessage] = [
            m for m in messages if not isinstance(m, SystemMessage)
        ]

        # --- 3. Apply the configurable message window -----------------------
        window = settings.RECENT_MESSAGES_WINDOW
        windowed = (
            filtered_messages[-window:]
            if len(filtered_messages) > window
            else filtered_messages
        )

        # --- 4. Assemble the final list ------------------------------------
        final_messages: List[BaseMessage] = [system_msg] + windowed

        # --- 5. Enforce the token budget ------------------------------------
        final_messages = _trim_to_budget(
            final_messages,
            budget_tokens=settings.CONTEXT_TOKEN_BUDGET,
            keep_first=1,  # always preserve the SystemMessage
        )

        logger.debug(
            "[ContextEngine] Context built: system=%d tokens | recent=%d msgs | total=%d tokens",
            len(_tokenizer.encode(system_content)),
            len(windowed),
            _count_tokens(final_messages),
        )

        return final_messages


# Module-level singleton
context_engine = ContextEngine()

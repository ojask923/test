import logging
from typing import Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from app.services.llm_factory import get_llm

logger = logging.getLogger(__name__)

class QueryRewriter:
    async def rewrite(
        self, 
        query: str, 
        messages: Sequence[BaseMessage], 
        provider: str, 
        model: str
    ) -> str:
        """Rewrite a search query to be standalone based on conversation history."""
        
        # Format the history into a string
        history_text = ""
        # We only need the last few messages for context to keep it fast
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        for m in recent_messages:
            if isinstance(m, HumanMessage):
                content = m.content
                if isinstance(content, list):
                    content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
                history_text += f"User: {content}\n"
            elif isinstance(m, AIMessage) and m.content:
                history_text += f"Assistant: {m.content}\n"

        if not history_text.strip():
            return query

        system_prompt = """You are an expert search query rewriter. 
Your task is to analyze the conversation history and a given search query.
If the query is already standalone and clear, output it exactly as is.
If the query relies on context (e.g. uses pronouns like 'it', 'they', 'this', 'that', or refers to previous topics implicitly), rewrite it into a single standalone query that captures the full context.
Do not answer the query. Do not add conversational filler.
Preserve important technical terminology, names, numbers, constraints, and user intent.
Return ONLY the final query text."""

        user_prompt = f"Conversation History:\n{history_text}\n\nCurrent Search Query: {query}"

        try:
            # We don't bind tools for the rewriter and use 0.0 temperature for consistency
            llm = get_llm(provider, model, temperature=0.0)
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ], config={"tags": ["query_rewriter"]})
            rewritten_query = response.content.strip()
            logger.info(f"[QueryRewriter] Original: '{query}' -> Rewritten: '{rewritten_query}'")
            return rewritten_query
        except Exception as exc:
            logger.warning(f"[QueryRewriter] Failed to rewrite query: {exc}")
            return query

query_rewriter = QueryRewriter()

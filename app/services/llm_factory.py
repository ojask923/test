import os
from typing import Optional, List, Any
from app.config import settings

def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    tools: Optional[List[Any]] = None,
):
    """Instantiate and configure an LLM based on provider."""
    provider = (provider or settings.DEFAULT_PROVIDER).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model or settings.DEFAULT_MODEL or "gpt-4o-mini",
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY or None,
        )
        return llm.bind_tools(tools) if tools else llm

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-flash",
            temperature=temperature,
            google_api_key=settings.GEMINI_API_KEY or None,
        )
        return llm.bind_tools(tools) if tools else llm

    elif provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model_name=model or settings.DEFAULT_MODEL or "openai/gpt-oss-120b",
            temperature=temperature,
            max_tokens=900,  # Stay under Groq's 1000 OTPM limit on free tier
            api_key=settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY"),
        )
        return llm.bind_tools(tools) if tools else llm

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model_name=model or "claude-3-5-sonnet-20241022",
            temperature=temperature,
            api_key=settings.ANTHROPIC_API_KEY or None,
        )
        return llm.bind_tools(tools) if tools else llm

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=model or "llama3.2",
            temperature=temperature,
        )
        return llm.bind_tools(tools) if tools else llm

    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model=model or settings.DEFAULT_MODEL or "poolside/laguna-s-2.1:free",
            temperature=temperature,
            api_key=settings.OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY"),
        )
        return llm.bind_tools(tools) if tools else llm

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. "
        "Supported: groq, ollama, openai, gemini, anthropic, openrouter"
    )

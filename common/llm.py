"""LLM factory. Returns an OpenAI-compatible chat model wired to OpenRouter."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — copy .env.example to .env")
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
        base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        temperature=temperature,
        # Without a cap OpenRouter reserves the model's full output window (e.g. 65k
        # tokens) against the credit balance and returns 402 on small accounts.
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "8000")),
    )

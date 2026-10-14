"""
app/agent/llm.py

Shared LLM factory for the agent nodes. Centralises the DeepSeek client
construction (model, API key, base URL from settings) so nodes never
hardcode provider details.
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel

from app.config import get_settings


def get_chat_llm(temperature: float = 0) -> ChatDeepSeek:
    """
    Return a ChatDeepSeek client configured from application settings.

    Args:
        temperature: Sampling temperature (0 for deterministic SQL output).
    """
    settings = get_settings()
    return ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key.get_secret_value(),
        api_base=settings.deepseek_base_url,
        temperature=temperature,
        # explicit client-level resilience (DESIGN_clarify.md DR-7): transient
        # network blips retry in the client instead of surfacing as failed
        # turns; a real outage fails fast instead of hanging each node
        request_timeout=settings.deepseek_timeout_seconds,
        max_retries=settings.deepseek_max_retries,
    )


def get_structured_llm(schema: type[BaseModel]):
    """
    Return a chat chain that parses into `schema`; invoke() returns the
    include_raw dict {"raw": AIMessage, "parsed": schema | None, ...}.

    DeepSeek thinking models (e.g. deepseek-flash) reject the
    tool_choice="required" that tool-calling structured output sends, so we
    use JSON mode instead. The caller MUST append json_format_instructions()
    to the prompt — it injects the exact schema and satisfies DeepSeek's
    requirement that the prompt contain the word "json".
    """
    return get_chat_llm().with_structured_output(schema, include_raw=True, method="json_mode")


def json_format_instructions(schema: type[BaseModel]) -> str:
    """Prompt instructions describing the JSON schema expected by get_structured_llm()."""
    return PydanticOutputParser(pydantic_object=schema).get_format_instructions()

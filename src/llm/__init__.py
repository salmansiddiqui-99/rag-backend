"""
LLM abstraction layer with OpenRouter API.

This module provides a unified interface for working with OpenRouter LLM provider
through an adapter pattern. The adapter implements an OpenAI-compatible interface.

Usage:
    from src.llm import create_llm_adapter, LLMCompletionParams, LLMMessage
    from src.config import settings

    # Create adapter (uses OpenRouter from settings)
    adapter = create_llm_adapter(settings)

    # Create completion
    params = LLMCompletionParams(
        messages=[
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="What is ROS 2?")
        ],
        model="mistralai/devstral-2512:free",
        temperature=0.7,
        stream=True
    )

    # Stream response
    for chunk in adapter.create_completion_stream(params):
        print(chunk.choices[0].delta.get("content", ""), end="")
"""

from .base import (
    BaseLLMAdapter,
    LLMMessage,
    LLMCompletionParams,
    LLMResponse,
    LLMStreamChunk,
    LLMAdapterError,
    LLMProviderError,
    LLMConfigurationError,
)

from .factory import (
    LLMFactory,
    create_llm_adapter,
)

# Export main classes and functions
__all__ = [
    # Base interfaces
    "BaseLLMAdapter",
    "LLMMessage",
    "LLMCompletionParams",
    "LLMResponse",
    "LLMStreamChunk",

    # Exceptions
    "LLMAdapterError",
    "LLMProviderError",
    "LLMConfigurationError",

    # Factory
    "LLMFactory",
    "create_llm_adapter",
]

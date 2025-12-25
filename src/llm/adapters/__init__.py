"""
LLM provider adapters.

This module contains adapter implementations for LLM providers.
All adapters implement the BaseLLMAdapter interface defined in llm.base.

NOTE: Only OpenRouter adapter is currently available and supported.
"""

# OpenRouter is the only supported provider
from .openrouter_adapter import OpenRouterAdapter

__all__ = ["OpenRouterAdapter"]

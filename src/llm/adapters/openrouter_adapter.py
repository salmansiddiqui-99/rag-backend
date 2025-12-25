"""
OpenRouter API adapter with OpenAI-compatible interface.

This adapter wraps OpenRouter API (which is OpenAI-compatible) to provide
seamless integration with models like Mistral, DeepSeek, and others.
"""

import logging
import time
import uuid
from typing import Iterator, Optional, List, Dict, Any

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "openai package is required for OpenRouterAdapter. "
        "Install it with: pip install openai"
    )

from ..base import (
    BaseLLMAdapter,
    LLMMessage,
    LLMCompletionParams,
    LLMResponse,
    LLMStreamChunk,
    LLMProviderError,
    LLMConfigurationError
)

logger = logging.getLogger(__name__)


class OpenRouterAdapter(BaseLLMAdapter):
    """
    Adapter for OpenRouter API with OpenAI-compatible interface.

    OpenRouter provides a unified API for accessing multiple LLM providers
    including Mistral, DeepSeek, OpenAI, Claude, and many others.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "mistralai/mistral-7b-instruct:free",
        base_url: str = "https://openrouter.ai/api/v1",
        **kwargs
    ):
        """
        Initialize OpenRouter adapter.

        Args:
            api_key: OpenRouter API key
            model: Model identifier (e.g., "mistralai/mistral-7b-instruct:free")
            base_url: OpenRouter API base URL
            **kwargs: Additional configuration
        """
        if not api_key:
            raise LLMConfigurationError("OpenRouter API key is required")

        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._kwargs = kwargs

        # Initialize OpenAI client pointed at OpenRouter
        try:
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            logger.info(f"Initialized OpenRouter adapter with model: {model}")
        except Exception as e:
            raise LLMConfigurationError(f"Failed to initialize OpenRouter client: {e}")

    def create_completion(
        self,
        params: LLMCompletionParams
    ) -> LLMResponse:
        """
        Create a non-streaming completion.

        Args:
            params: Unified completion parameters

        Returns:
            LLMResponse in OpenAI-compatible format
        """
        try:
            # Convert messages to OpenAI format
            # Handle both LLMMessage objects and plain dictionaries
            messages = []
            for msg in params.messages:
                if isinstance(msg, dict):
                    # Already a dictionary
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                else:
                    # LLMMessage object with attributes
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })

            # Create completion request
            response = self._client.chat.completions.create(
                model=params.model or self._model,
                messages=messages,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
            )

            # Extract response content
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"

            # Generate completion ID
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            return LLMResponse(
                id=completion_id,
                object="chat.completion",
                created=created,
                model=params.model or self._model,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )

        except Exception as e:
            logger.error(f"OpenRouter completion error: {e}")
            raise LLMProviderError(f"OpenRouter API error: {str(e)}")

    def create_completion_stream(
        self,
        params: LLMCompletionParams
    ) -> Iterator[LLMStreamChunk]:
        """
        Create a streaming completion.

        Args:
            params: Unified completion parameters

        Yields:
            LLMStreamChunk objects with token-by-token responses
        """
        try:
            # Convert messages to OpenAI format
            # Handle both LLMMessage objects and plain dictionaries
            messages = []
            for msg in params.messages:
                if isinstance(msg, dict):
                    # Already a dictionary
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                else:
                    # LLMMessage object with attributes
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })

            # Create streaming request
            stream = self._client.chat.completions.create(
                model=params.model or self._model,
                messages=messages,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
                stream=True,
            )

            # Generate chunk ID (same for all chunks in this stream)
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            for chunk in stream:
                # Extract delta content if present
                delta = {}
                if chunk.choices[0].delta.content:
                    delta["content"] = chunk.choices[0].delta.content

                # Extract finish reason if present
                finish_reason = chunk.choices[0].finish_reason

                yield LLMStreamChunk(
                    id=chunk_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=params.model or self._model,
                    choices=[
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish_reason,
                        }
                    ]
                )

        except Exception as e:
            logger.error(f"OpenRouter streaming error: {e}")
            raise LLMProviderError(f"OpenRouter streaming error: {str(e)}")

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for the given text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count (approximately 4 chars per token)
        """
        try:
            # OpenRouter doesn't have a dedicated token counting API,
            # so we use a rough estimation based on character count
            # Most models use roughly 1 token per 4 characters
            return max(1, len(text) // 4)
        except Exception:
            # Fallback to character-based estimate
            return max(1, len(text) // 4)

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "openrouter"

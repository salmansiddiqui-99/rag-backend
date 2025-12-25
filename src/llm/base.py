"""
Abstract base interface for LLM providers.

This module defines the core interfaces and data models for LLM adapters,
providing a unified API for LLM providers. Currently supports OpenRouter.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LLMMessage:
    """
    Unified message format compatible with OpenAI's message structure.

    Attributes:
        role: Message role ("system", "user", "assistant", "function", "tool")
        content: Message content text
        name: Optional name for function/tool messages
        function_call: Optional function call data (deprecated, use tool_calls)
        tool_calls: Optional list of tool call objects
    """
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class LLMCompletionParams:
    """
    Unified completion parameters compatible with OpenAI's API.

    Attributes:
        messages: List of messages in the conversation
        model: Model name to use
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        stream: Whether to stream the response
        tools: Optional list of available tools/functions
        tool_choice: How to choose tools ("auto", "none", or specific tool)
        top_p: Nucleus sampling parameter
        frequency_penalty: Penalty for token frequency
        presence_penalty: Penalty for token presence
        stop: List of stop sequences
        n: Number of completions to generate
        user: Optional user identifier
    """
    messages: List[LLMMessage]
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = "auto"
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    n: int = 1
    user: Optional[str] = None


@dataclass
class LLMResponse:
    """
    Unified response format compatible with OpenAI's ChatCompletion.

    Attributes:
        id: Unique completion ID
        object: Object type ("chat.completion")
        created: Unix timestamp of creation
        model: Model used for completion
        choices: List of completion choices
        usage: Token usage statistics
        system_fingerprint: Optional system fingerprint
    """
    id: str
    object: str  # "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    system_fingerprint: Optional[str] = None


@dataclass
class LLMStreamChunk:
    """
    Unified streaming chunk format compatible with OpenAI's ChatCompletionChunk.

    Attributes:
        id: Unique chunk ID (same for all chunks in a stream)
        object: Object type ("chat.completion.chunk")
        created: Unix timestamp of creation
        model: Model used for completion
        choices: List of delta choices
        system_fingerprint: Optional system fingerprint
    """
    id: str
    object: str  # "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    system_fingerprint: Optional[str] = None


class BaseLLMAdapter(ABC):
    """
    Abstract base class for LLM provider adapters.

    This class defines the interface that all LLM adapters must implement.
    Currently supports OpenRouter for LLM operations.
    """

    @abstractmethod
    def __init__(self, api_key: str, model: str, **kwargs):
        """
        Initialize the adapter with provider-specific credentials.

        Args:
            api_key: API key for the LLM provider
            model: Default model name to use
            **kwargs: Provider-specific additional configuration
        """
        pass

    @abstractmethod
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

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If API call fails
        """
        pass

    @abstractmethod
    def create_completion_stream(
        self,
        params: LLMCompletionParams
    ) -> Iterator[LLMStreamChunk]:
        """
        Create a streaming completion.

        Args:
            params: Unified completion parameters

        Yields:
            LLMStreamChunk in OpenAI-compatible format

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If API call fails
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for the given text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Get the current model name.

        Returns:
            Model name string
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the provider name.

        Returns:
            Provider name string (e.g., "openrouter")
        """
        pass


class LLMAdapterError(Exception):
    """Base exception for LLM adapter errors."""
    pass


class LLMProviderError(LLMAdapterError):
    """Exception raised when the LLM provider returns an error."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class LLMConfigurationError(LLMAdapterError):
    """Exception raised when adapter configuration is invalid."""
    pass

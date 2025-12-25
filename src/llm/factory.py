"""
LLM provider factory for creating adapters based on configuration.

This factory creates OpenRouter adapters for LLM operations.
OpenRouter is the only supported provider.
"""

from typing import Optional, Dict, Any
import logging

from .base import BaseLLMAdapter, LLMConfigurationError
from .adapters.openrouter_adapter import OpenRouterAdapter

# NOTE: Only OpenRouter is supported.

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory for creating LLM adapters.

    IMPORTANT: Only OpenRouter is supported. All other providers have been removed.
    """

    @staticmethod
    def create_from_settings(settings: Any) -> BaseLLMAdapter:
        """
        Create an LLM adapter from application settings.

        IMPORTANT: OpenRouter is the REQUIRED provider. Other providers are not supported.

        Args:
            settings: Application settings object with LLM configuration

        Returns:
            Configured OpenRouter LLM adapter instance

        Raises:
            LLMConfigurationError: If OpenRouter is not properly configured
        """
        provider = getattr(settings, "LLM_PROVIDER", "openrouter").lower()

        logger.info(f"Creating LLM adapter with provider: {provider}")

        # Only OpenRouter is supported
        if provider in ["openrouter", "auto"]:
            # For "auto", detect and use OpenRouter
            if provider == "auto":
                provider = LLMFactory._auto_detect_provider(settings)
                logger.info(f"Auto-detected provider: {provider}")

            return LLMFactory.create_openrouter_adapter(settings)
        else:
            raise LLMConfigurationError(
                f"Unsupported LLM provider: {provider}. "
                f"Only 'openrouter' is supported. "
                f"Please set LLM_PROVIDER='openrouter' and configure OPENROUTER_API_KEY."
            )

    @staticmethod
    def create_openrouter_adapter(settings: Any) -> BaseLLMAdapter:
        """
        Create an OpenRouter adapter from settings.

        Args:
            settings: Application settings with OPENROUTER_API_KEY and OPENROUTER_MODEL

        Returns:
            Configured OpenRouterAdapter instance

        Raises:
            LLMConfigurationError: If OpenRouter configuration is missing
        """
        api_key = getattr(settings, "OPENROUTER_API_KEY", None)
        if not api_key:
            raise LLMConfigurationError(
                "OPENROUTER_API_KEY is required for OpenRouter provider. "
                "Set it in your .env file or environment variables."
            )

        # Get model name, with fallback
        model = getattr(settings, "OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
        base_url = getattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        # Get optional configuration
        temperature = getattr(settings, "OPENAI_TEMPERATURE", 0.7)
        max_tokens = getattr(settings, "OPENAI_MAX_TOKENS", 2000)

        logger.info(
            f"Creating OpenRouterAdapter with model={model}, "
            f"temperature={temperature}, max_tokens={max_tokens}"
        )

        return OpenRouterAdapter(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )


    @staticmethod
    def create_adapter(
        provider: str,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> BaseLLMAdapter:
        """
        Create an LLM adapter with explicit parameters.

        IMPORTANT: Only OpenRouter is supported.

        This is a lower-level method for creating adapters when you have
        explicit configuration values instead of a settings object.

        Args:
            provider: Provider name ("openrouter" only)
            api_key: API key for OpenRouter
            model: Optional model name (uses OpenRouter default if None)
            base_url: Optional base URL (defaults to OpenRouter API URL)
            **kwargs: Additional provider-specific configuration

        Returns:
            Configured OpenRouter adapter instance

        Raises:
            LLMConfigurationError: If configuration is invalid
        """
        provider = provider.lower()

        if provider == "openrouter":
            model = model or "mistralai/devstral-2512:free"
            base_url = base_url or "https://openrouter.ai/api/v1"
            return OpenRouterAdapter(api_key=api_key, model=model, base_url=base_url, **kwargs)
        else:
            raise LLMConfigurationError(
                f"Unsupported provider: {provider}. Only 'openrouter' is supported."
            )

    @staticmethod
    def _auto_detect_provider(settings: Any) -> str:
        """
        Auto-detect the LLM provider based on available API keys.

        OpenRouter is the REQUIRED and only supported provider.
        This method ensures we use OpenRouter for all LLM operations.

        Raises:
            LLMConfigurationError: If OpenRouter is not configured
        """
        openrouter_key = getattr(settings, "OPENROUTER_API_KEY", None)
        if openrouter_key:
            logger.info("Using OpenRouter as LLM provider")
            return "openrouter"

        raise LLMConfigurationError(
            "OPENROUTER_API_KEY is required and not configured. "
            "Please set OPENROUTER_API_KEY in your environment variables or .env file. "
            "Get your API key from: https://openrouter.ai/"
        )

    @staticmethod
    def get_available_providers(settings: Any) -> list[str]:
        """
        Get list of available providers based on configured API keys.

        Currently, only OpenRouter is supported.

        Args:
            settings: Application settings object

        Returns:
            List of available provider names (only 'openrouter' if configured)
        """
        providers = []

        if getattr(settings, "OPENROUTER_API_KEY", None):
            providers.append("openrouter")

        return providers


# Convenience function for quick adapter creation
def create_llm_adapter(settings: Any) -> BaseLLMAdapter:
    """
    Convenience function to create an LLM adapter from settings.

    This is a shortcut for LLMFactory.create_from_settings(settings).

    Args:
        settings: Application settings object

    Returns:
        Configured LLM adapter instance
    """
    return LLMFactory.create_from_settings(settings)

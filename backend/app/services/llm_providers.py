"""
Dynamic LLM Provider System supporting multiple backends.

This module provides a unified interface for different LLM providers:
- ChatOllama (default)
- ChatGroq
- ChatOpenAI
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import os
import requests
try:
    from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
except ImportError:
    # Fallback for type hints if langchain is not available
    BaseMessage = Any
    HumanMessage = Any
    SystemMessage = Any
    AIMessage = Any


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    async def generate_response(self, messages: List[Any]) -> str:
        """Generate a response from the given messages."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass


class ChatOllamaProvider(LLMProvider):
    """Ollama LLM provider implementation."""

    def __init__(self, model_name: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self._chat_model = None

    @property
    def provider_name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        """Check if Ollama service is running and model is available."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False

            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any(self.model_name == name or name.startswith(self.model_name) for name in model_names)

        except (requests.exceptions.RequestException, ValueError):
            return False

    async def generate_response(self, messages: List[Any]) -> str:
        """Generate response using Ollama."""
        if not self._chat_model:
            try:
                from langchain_ollama import ChatOllama
                self._chat_model = ChatOllama(
                    base_url=self.base_url,
                    model=self.model_name,
                    temperature=0.9,
                    **self.kwargs
                )
            except ImportError:
                raise RuntimeError("langchain_ollama not installed. Install with: pip install langchain_ollama")

        response = await self._chat_model.ainvoke(messages)
        return response.content.strip()


class ChatOpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        #self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._chat_model = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    async def generate_response(self, messages: List[Any]) -> str:
        """Generate response using OpenAI."""
        if not self._chat_model:
            try:
                from langchain_openai import ChatOpenAI
                if not self.api_key:
                    raise RuntimeError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")

                self._chat_model = ChatOpenAI(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=0.9,
                    **self.kwargs
                )
            except ImportError:
                raise RuntimeError("langchain_openai not installed. Install with: pip install langchain_openai")

        response = await self._chat_model.ainvoke(messages)
        return response.content.strip()


class ChatGroqProvider(LLMProvider):
    """Groq LLM provider implementation."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        # ✅ read from arg or env
        self.api_key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()
        #self.api_key ="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".strip()
        self._chat_model = None

    @property
    def provider_name(self) -> str:
        return "groq"

    def is_available(self) -> bool:
        """Check if Groq API key is configured."""
        return bool(self.api_key)

    async def generate_response(self, messages: List[Any]) -> str:
        """Generate response using Groq."""
        if not self._chat_model:
            try:
                from langchain_groq import ChatGroq
                if not self.api_key:
                    raise RuntimeError("Groq API key not configured. Set GROQ_API_KEY environment variable.")
                self._chat_model = ChatGroq(
                    model=self.model_name,
                    api_key=self.api_key,   # ✅ use the resolved key, do NOT hardcode
                    temperature=0.9,
                    **self.kwargs
                )
            except ImportError:
                raise RuntimeError("langchain_groq not installed. Install with: pip install langchain_groq")

        response = await self._chat_model.ainvoke(messages)
        return response.content.strip()



class LLMProviderFactory:
    """Factory class for creating LLM providers."""

    # Supported providers mapping
    PROVIDERS = {
        "ollama": ChatOllamaProvider,
        "openai": ChatOpenAIProvider,
        "groq": ChatGroqProvider,
    }

    @classmethod
    def create_provider(
        cls,
        provider_name: Optional[str] = None,
        model_name: str = "gemma3:1b",
        **kwargs
    ) -> LLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider_name: Name of the provider ("ollama", "openai", "groq")
            model_name: Name of the model to use
            **kwargs: Additional arguments for the provider

        Returns:
            LLMProvider instance

        Raises:
            ValueError: If provider is not supported
        """
        # Auto-detect provider from environment or use default
        if not provider_name:
            if os.getenv("OPENAI_API_KEY"):
                provider_name = "openai"
            elif os.getenv("GROQ_API_KEY"):
                provider_name = "groq"
            else:
                provider_name = "ollama"  # Default

        if provider_name not in cls.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider_name}. Supported: {list(cls.PROVIDERS.keys())}")

        provider_class = cls.PROVIDERS[provider_name]
        return provider_class(model_name=model_name, **kwargs)

    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """Get availability status of all providers."""
        status = {}
        for name in cls.PROVIDERS.keys():
            try:
                provider = cls.create_provider(name)
                status[name] = provider.is_available()
            except Exception:
                status[name] = False
        return status

    @classmethod
    def list_supported_providers(cls) -> List[str]:
        """List all supported provider names."""
        return list(cls.PROVIDERS.keys())


def create_llm_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """
    Convenience function to create an LLM provider.

    Args:
        provider: Provider name ("ollama", "openai", "groq")
        model: Model name
        **kwargs: Additional provider-specific arguments

    Returns:
        LLMProvider instance
    """
    return LLMProviderFactory.create_provider(
        provider_name=provider,
        model_name=model or "gemma3:1b",
        **kwargs
    )
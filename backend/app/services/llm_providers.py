"""
Dynamic LLM Provider System supporting multiple backends.

This module provides a unified interface for different LLM providers:
- ChatOllama (default)
- ChatGroq
- ChatOpenAI
- ChatHuggingFace (local models)
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import os
import requests
import asyncio
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
                print(f"⚠️  Ollama server not responding at {self.base_url}")
                return False

            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            is_model_available = any(self.model_name == name or name.startswith(self.model_name) for name in model_names)

            if not is_model_available:
                print(f"⚠️  Model '{self.model_name}' not found in Ollama. Available models: {model_names}")

            return is_model_available

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Cannot connect to Ollama server at {self.base_url}: {e}")
            return False
        except ValueError as e:
            print(f"⚠️  Error parsing Ollama response: {e}")
            return False

    async def generate_response(self, messages: List[Any]) -> str:
        """Generate response using Ollama."""
        if not self._chat_model:
            try:
                from langchain_ollama import ChatOllama
                from backend.app.services.llm_config import model_config
                self._chat_model = ChatOllama(
                    base_url=self.base_url,
                    model=self.model_name,
                    temperature=model_config.temperature,
                    **{k: v for k, v in self.kwargs.items() if k not in ['base_url']}
                )
            except ImportError:
                raise RuntimeError("langchain_ollama not installed. Install with: pip install langchain_ollama")

        response = await self._chat_model.ainvoke(messages)
        return response.content.strip()


class ChatOpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
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

                from backend.app.services.llm_config import model_config
                self._chat_model = ChatOpenAI(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=model_config.temperature,
                    **{k: v for k, v in self.kwargs.items() if k not in ['api_key']}
                )
            except ImportError:
                raise RuntimeError("langchain_openai not installed. Install with: pip install langchain_openai")

        response = await self._chat_model.ainvoke(messages)
        return response.content.strip()


class ChatGroqProvider(LLMProvider):
    """Groq LLM provider implementation."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self._chat_model = None

    @property
    def provider_name(self) -> str:
        return "groq"

    def is_available(self) -> bool:
        """Check if Groq API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    async def generate_response(self, messages: List[Any]) -> str:
        """Generate response using Groq."""
        if not self._chat_model:
            try:
                from langchain_groq import ChatGroq
                if not self.api_key:
                    raise RuntimeError("Groq API key not configured. Set GROQ_API_KEY environment variable.")

                from backend.app.services.llm_config import model_config
                self._chat_model = ChatGroq(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=model_config.temperature,
                    **{k: v for k, v in self.kwargs.items() if k not in ['api_key']}
                )
            except ImportError:
                raise RuntimeError("langchain_groq not installed. Install with: pip install langchain_groq")

        response = await self._chat_model.ainvoke(messages)
        return response.content.strip()


class ChatHuggingFaceProvider(LLMProvider):
    """HuggingFace local LLM provider implementation."""

    def __init__(self, model_name: str, model_path: Optional[str] = None, device: str = "auto", preload_model: bool = False, **kwargs):
        super().__init__(model_name, **kwargs)
        # Default to SmolLM3-3B if no specific model path provided
        default_model = "HuggingFaceTB/SmolLM3-3B" if not model_path else model_path
        self.model_path = model_path or default_model
        self.device = device
        self.preload_model = preload_model or os.getenv("HUGGINGFACE_PRELOAD_MODEL", "false").lower() == "true"
        self._chat_model = None
        self._tokenizer = None
        self._model_loading = False
        self._model_load_error = None

        # Preload model if requested
        if self.preload_model:
            try:
                self._load_model()
            except Exception as e:
                print(f"Warning: Failed to preload HuggingFace model {self.model_path}: {e}")
                self._model_load_error = str(e)

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def is_available(self) -> bool:
        """Check if HuggingFace model is available and properly configured."""
        try:
            # Check if transformers is installed
            import transformers
            # Check if model path exists or is accessible (without loading)
            try:
                from transformers import AutoConfig
                AutoConfig.from_pretrained(self.model_path, timeout=5)
                return True
            except Exception:
                # Model not accessible, but transformers is installed
                return True
        except ImportError:
            return False

    def _load_model(self) -> None:
        """Load the HuggingFace model with timeout and error handling."""
        if self._model_loading:
            # Prevent concurrent loading attempts
            return

        self._model_loading = True
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch

            print(f"Loading HuggingFace model: {self.model_path}")

            # Initialize tokenizer and model with timeout handling
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    timeout=30  # 30 second timeout for downloads
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer for {self.model_path}: {e}")

            # Set pad token if not exists
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    dtype="auto",
                    device_map="auto" if torch.cuda.is_available() else None,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model {self.model_path}: {e}")

            from backend.app.services.llm_config import model_config
            # Create pipeline for text generation
            self._chat_model = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                temperature=model_config.temperature,
                max_new_tokens=512,
                pad_token_id=self._tokenizer.eos_token_id,
                **self.kwargs
            )

            print(f"✅ Successfully loaded HuggingFace model: {self.model_path}")

        except Exception as e:
            self._model_load_error = str(e)
            print(f"❌ Failed to load model {self.model_path}: {e}")
            raise
        finally:
            self._model_loading = False

    async def generate_response(self, messages: List[Any]) -> str:
        """Generate response using HuggingFace local model."""
        # Check if model failed to load during initialization
        if self._model_load_error:
            raise RuntimeError(f"Model failed to load during initialization: {self._model_load_error}")

        # Load model if not already loaded
        if not self._chat_model or not self._tokenizer:
            try:
                self._load_model()
            except Exception as e:
                raise RuntimeError(f"Failed to load HuggingFace model {self.model_path}: {e}")

        # Convert messages to prompt format
        if len(messages) >= 2:
            system_message = messages[0].content if hasattr(messages[0], 'content') else str(messages[0])
            user_message = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
            prompt = f"System: {system_message}\n\nHuman: {user_message}\n\nAssistant:"
        else:
            prompt = str(messages[0]) if messages else ""

        try:
            # Generate response with timeout
            loop = asyncio.get_event_loop()
            outputs = await asyncio.wait_for(
                loop.run_in_executor(None, self._chat_model, prompt),
                timeout=30  # 30 second timeout for generation
            )

            if isinstance(outputs, list) and len(outputs) > 0:
                generated_text = outputs[0].get('generated_text', '')
            else:
                generated_text = str(outputs) if outputs else ""

            # Extract only the assistant's response (remove the prompt)
            if "Assistant:" in generated_text:
                response = generated_text.split("Assistant:")[-1].strip()
            else:
                response = generated_text.replace(prompt, "").strip()

            return response

        except asyncio.TimeoutError:
            raise RuntimeError(f"Model generation timed out for {self.model_path}. The model may be overloaded or the request too complex.")
        except Exception as e:
            raise RuntimeError(f"Error generating response with HuggingFace model {self.model_path}: {e}")


class LLMProviderFactory:
    """Factory class for creating LLM providers."""

    # Supported providers mapping
    PROVIDERS = {
        "ollama": ChatOllamaProvider,
        "openai": ChatOpenAIProvider,
        "groq": ChatGroqProvider,
        "huggingface": ChatHuggingFaceProvider,
    }

    @classmethod
    def create_provider(
        cls,
        provider_name: Optional[str] = None,
        model_name: str = "HuggingFaceTB/SmolLM3-3B",
        preload_model: bool = False,
        **kwargs
    ) -> LLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider_name: Name of the provider ("ollama", "openai", "groq", "huggingface")
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
            elif os.getenv("OLLAMA_BASE_URL"):
                provider_name = "ollama"
            elif os.getenv("HUGGINGFACE_MODEL_PATH"):
                provider_name = "huggingface"
            else:
                # Default to ollama if available, otherwise openai if key exists, fallback to huggingface
                provider_name = "ollama"  # Changed from huggingface to avoid server startup hanging

        if provider_name not in cls.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider_name}. Supported: {list(cls.PROVIDERS.keys())}")

        provider_class = cls.PROVIDERS[provider_name]

        # Filter out provider-specific parameters that shouldn't go to LangChain
        # Only HuggingFace provider uses model_path and device parameters
        if provider_name == 'huggingface':
            filtered_kwargs = kwargs
        else:
            # For other providers, filter out HuggingFace-specific parameters
            filtered_kwargs = {k: v for k, v in kwargs.items()
                              if k not in ['preload_model', 'model_path', 'device']}

        # Only pass preload_model to HuggingFace provider
        if provider_name == 'huggingface':
            return provider_class(
                model_name=model_name,
                preload_model=preload_model,
                **filtered_kwargs
            )
        else:
            return provider_class(
                model_name=model_name,
                **filtered_kwargs
            )

    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """Get availability status of all providers."""
        status = {}
        for name in cls.PROVIDERS.keys():
            try:
                # For HuggingFace, just check if transformers is available without creating provider
                if name == "huggingface":
                    try:
                        import transformers
                        status[name] = True
                    except ImportError:
                        status[name] = False
                else:
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
    preload_model: bool = False,
    **kwargs
) -> LLMProvider:
    """
    Convenience function to create an LLM provider.

    Args:
        provider: Provider name ("ollama", "openai", "groq", "huggingface")
        model: Model name
        **kwargs: Additional provider-specific arguments

    Returns:
        LLMProvider instance
    """
    return LLMProviderFactory.create_provider(
        provider_name=provider,
        model_name=model or "HuggingFaceTB/SmolLM3-3B",
        preload_model=preload_model,
        **kwargs
    )
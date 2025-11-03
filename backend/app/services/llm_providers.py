# from abc import ABC, abstractmethod
# from typing import List, Dict, Any, Optional, Union
# import os
# import requests
# import asyncio
# from ..settings.settings import settings
# try:
#     from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
# except ImportError:
#     # Fallback for type hints if langchain is not available
#     BaseMessage = Any
#     HumanMessage = Any
#     SystemMessage = Any
#     AIMessage = Any


# class LLMProvider(ABC):
#     """Abstract base class for LLM providers."""

#     def __init__(self, model_name: str, **kwargs):
#         self.model_name = model_name
#         self.kwargs = kwargs

#     @abstractmethod
#     async def generate_response(self, messages: List[Any]) -> str:
#         """Generate a response from the given messages."""
#         pass
    
#     async def agenerate(self, messages: List[Any]) -> str:
#         """Alias for generate_response for compatibility with pipeline nodes."""
#         return await self.generate_response(messages)

#     @abstractmethod
#     def is_available(self) -> bool:
#         """Check if the provider is available and properly configured."""
#         pass

#     @property
#     @abstractmethod
#     def provider_name(self) -> str:
#         """Return the provider name."""
#         pass

#     def _extract_content(self, response: Any) -> str:
#         """Extract content from various LLM response formats."""
#         if hasattr(response, 'content') and response.content is not None:
#             content = response.content
#             if isinstance(content, str):
#                 return content.strip()
#             elif isinstance(content, list) and content:
#                 first_item = content
#                 if isinstance(first_item, dict) and 'text' in first_item and isinstance(first_item['text'], str):
#                     return first_item['text'].strip()
#                 elif isinstance(first_item, str):
#                     return first_item.strip()
#         return str(response).strip() # Fallback to string conversion


# class ChatOllamaProvider(LLMProvider):
#     """Ollama LLM provider implementation."""

#     def __init__(self, model_name: str, base_url: Optional[str] = None, **kwargs):
#         super().__init__(model_name, **kwargs)
#         self.base_url = base_url or (settings.model.ollama_base_url if settings.model else "http://127.0.0.1:11434")
#         self._chat_model = None

#     @property
#     def provider_name(self) -> str:
#         return "ollama"

#     def is_available(self) -> bool:
#         """Check if Ollama service is running and model is available."""
#         try:
#             # Check if Ollama is running
#             response = requests.get(f"{self.base_url}/api/tags", timeout=5)
#             if response.status_code != 200:
#                 print(f"⚠️  Ollama server not responding at {self.base_url}")
#                 return False

#             # Check if model is available
#             models = response.json().get("models", [])
#             model_names = [m.get("name", "") for m in models]
#             is_model_available = any(self.model_name == name or name.startswith(self.model_name) for name in model_names)

#             if not is_model_available:
#                 print(f"⚠️  Model '{self.model_name}' not found in Ollama. Available models: {model_names}")

#             return is_model_available

#         except requests.exceptions.RequestException as e:
#             print(f"⚠️  Cannot connect to Ollama server at {self.base_url}: {e}")
#             return False
#         except ValueError as e:
#             print(f"⚠️  Error parsing Ollama response: {e}")
#             return False

#     async def generate_response(self, messages: List[Any]) -> str:
#         """Generate response using Ollama."""
#         if not self._chat_model:
#             try:
#                 from langchain_ollama import ChatOllama
#                 # Use the compatibility layer for gradual migration
#                 model_temperature = settings.model.temperature if settings.model else 0.85
#                 self._chat_model = ChatOllama(
#                     base_url=self.base_url,
#                     model=self.model_name,
#                     temperature=model_temperature,
#                     **{k: v for k, v in self.kwargs.items() if k not in ['base_url']}
#                 )
#                 print(f"✅ Ollama chat model created: {self.model_name}")
#             except ImportError:
#                 raise RuntimeError("langchain_ollama not installed. Install with: pip install langchain_ollama")

#         response = await self._chat_model.ainvoke(messages)
#         return self._extract_content(response)


# class ChatOpenAIProvider(LLMProvider):
#     """OpenAI LLM provider implementation."""

#     def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
#         super().__init__(model_name, **kwargs)
#         #self.api_key = api_key or os.getenv("OPENAI_API_KEY")
#         self.api_key = api_key or os.getenv("OPENAI_API_KEY")
#         self._chat_model = None

#     @property
#     def provider_name(self) -> str:
#         return "openai"

#     def is_available(self) -> bool:
#         """Check if OpenAI API key is configured."""
#         return bool(self.api_key and self.api_key.strip())

#     async def generate_response(self, messages: List[Any]) -> str:
#         """Generate response using OpenAI."""
#         if not self._chat_model:
#             try:
#                 from langchain_openai import ChatOpenAI
#                 if not self.api_key:
#                     raise RuntimeError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")

#                 # Use the compatibility layer for gradual migration
#                 model_temperature = settings.model.temperature if settings.model else 1.0
#                 self._chat_model = ChatOpenAI(
#                     model=self.model_name,
#                     api_key=self.api_key,
#                     temperature=model_temperature,
#                     **{k: v for k, v in self.kwargs.items() if k not in ['api_key']}
#                 )
#             except ImportError:
#                 raise RuntimeError("langchain_openai not installed. Install with: pip install langchain_openai")

#         response = await self._chat_model.ainvoke(messages)
#         return self._extract_content(response)


# class ChatGroqProvider(LLMProvider):
#     """Groq LLM provider implementation."""

#     def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
#         super().__init__(model_name, **kwargs)
#         # ✅ read from arg or env
#         self.api_key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()
#         #self.api_key ="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".strip()
#         self._chat_model = None

#     @property
#     def provider_name(self) -> str:
#         return "groq"

#     def is_available(self) -> bool:
#         """Check if Groq API key is configured."""
#         return bool(self.api_key)

#     async def generate_response(self, messages: List[Any]) -> str:
#         """Generate response using Groq."""
#         if not self._chat_model:
#             try:
#                 from langchain_groq import ChatGroq
#                 if not self.api_key:
#                     raise RuntimeError("Groq API key not configured. Set GROQ_API_KEY environment variable.")
# <<<<<<< HEAD
#                 self._chat_model = ChatGroq(
#                     model=self.model_name,
#                     api_key=self.api_key,   # ✅ use the resolved key, do NOT hardcode
#                     temperature=0.9,
#                     **self.kwargs
# =======

#                 # Use the compatibility layer for gradual migration
#                 model_temperature = settings.model.temperature if settings.model else 1.0
#                 self._chat_model = ChatGroq(
#                     model=self.model_name,
#                     api_key=self.api_key,
#                     temperature=model_temperature,
#                     **{k: v for k, v in self.kwargs.items() if k not in ['api_key']}
# >>>>>>> origin/main
#                 )
#             except ImportError:
#                 raise RuntimeError("langchain_groq not installed. Install with: pip install langchain_groq")

#         response = await self._chat_model.ainvoke(messages)
#         return self._extract_content(response)


# class ChatHuggingFaceProvider(LLMProvider):
#     """HuggingFace local LLM provider implementation."""

#     def __init__(self, model_name: str, model_path: Optional[str] = None, device: str = "auto", preload_model: bool = False, **kwargs):
#         super().__init__(model_name, **kwargs)
#         # Default to SmolLM3-3B if no specific model path provided
#         default_model = "HuggingFaceTB/SmolLM3-3B" if not model_path else model_path
#         self.model_path = model_path or default_model
#         self.device = device
#         self.preload_model = preload_model or os.getenv("HUGGINGFACE_PRELOAD_MODEL", "false").lower() == "true"
#         self._chat_model = None
#         self._tokenizer = None
#         self._model_loading = False
#         self._model_load_error = None

#         # Preload model if requested
#         if self.preload_model:
#             try:
#                 self._load_model()
#             except Exception as e:
#                 print(f"Warning: Failed to preload HuggingFace model {self.model_path}: {e}")
#                 self._model_load_error = str(e)

#     @property
#     def provider_name(self) -> str:
#         return "huggingface"

#     def is_available(self) -> bool:
#         """Check if HuggingFace model is available and properly configured."""
#         try:
#             # Check if transformers is installed
#             import transformers
#             # Check if model path exists or is accessible (without loading)
#             try:
#                 from transformers import AutoConfig
#                 AutoConfig.from_pretrained(self.model_path, timeout=5)
#                 return True
#             except Exception:
#                 # Model not accessible, but transformers is installed
#                 return True
#         except ImportError:
#             return False

#     def _load_model(self) -> None:
#         """Load the HuggingFace model with timeout and error handling."""
#         if self._model_loading:
#             # Prevent concurrent loading attempts
#             return

#         self._model_loading = True
#         try:
#             from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#             import torch

#             print(f"Loading HuggingFace model: {self.model_path}")

#             # Initialize tokenizer and model with timeout handling
#             try:
#                 self._tokenizer = AutoTokenizer.from_pretrained(
#                     self.model_path,
#                     timeout=30  # 30 second timeout for downloads
#                 )
#             except Exception as e:
#                 raise RuntimeError(f"Failed to load tokenizer for {self.model_path}: {e}")

#             # Set pad token if not exists
#             if self._tokenizer.pad_token is None:
#                 self._tokenizer.pad_token = self._tokenizer.eos_token

#             try:
#                 self._model = AutoModelForCausalLM.from_pretrained(
#                     self.model_path,
#                     dtype="auto",
#                     device_map="auto" if torch.cuda.is_available() else None,
#                     pad_token_id=self._tokenizer.eos_token_id
#                 )
#             except Exception as e:
#                 raise RuntimeError(f"Failed to load model {self.model_path}: {e}")

#             # Use the compatibility layer for gradual migration
#             model_temperature = settings.model.temperature if settings.model else 1.0
#             # Create pipeline for text generation
#             self._chat_model = pipeline(
#                 "text-generation",
#                 model=self._model,
#                 tokenizer=self._tokenizer,
#                 temperature=model_temperature,
#                 max_new_tokens=512,
#                 pad_token_id=self._tokenizer.eos_token_id,
#                 **self.kwargs
#             )

#             print(f"✅ Successfully loaded HuggingFace model: {self.model_path}")

#         except Exception as e:
#             self._model_load_error = str(e)
#             print(f"❌ Failed to load model {self.model_path}: {e}")
#             raise
#         finally:
#             self._model_loading = False

#     async def generate_response(self, messages: List[Any]) -> str:
#         """Generate response using HuggingFace local model."""
#         # Check if model failed to load during initialization
#         if self._model_load_error:
#             raise RuntimeError(f"Model failed to load during initialization: {self._model_load_error}")

#         # Load model if not already loaded
#         if not self._chat_model or not self._tokenizer:
#             try:
#                 self._load_model()
#             except Exception as e:
#                 raise RuntimeError(f"Failed to load HuggingFace model {self.model_path}: {e}")

#         # Convert messages to prompt format
#         if len(messages) >= 2:
#             system_message = messages[0].content if hasattr(messages[0], 'content') else str(messages[0])
#             user_message = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
#             prompt = f"System: {system_message}\n\nHuman: {user_message}\n\nAssistant:"
#         else:
#             prompt = str(messages[0]) if messages else ""

#         try:
#             # Generate response with timeout
#             loop = asyncio.get_event_loop()
#             outputs = await asyncio.wait_for(
#                 loop.run_in_executor(None, self._chat_model, prompt),
#                 timeout=30  # 30 second timeout for generation
#             )

#             if isinstance(outputs, list) and len(outputs) > 0:
#                 generated_text = outputs[0].get('generated_text', '')
#             else:
#                 generated_text = str(outputs) if outputs else ""

#             # Extract only the assistant's response (remove the prompt)
#             if "Assistant:" in generated_text:
#                 response = generated_text.split("Assistant:")[-1].strip()
#             else:
#                 response = generated_text.replace(prompt, "").strip()

#             return response

#         except asyncio.TimeoutError:
#             raise RuntimeError(f"Model generation timed out for {self.model_path}. The model may be overloaded or the request too complex.")
#         except Exception as e:
#             raise RuntimeError(f"Error generating response with HuggingFace model {self.model_path}: {e}")



# class LLMProviderFactory:
#     """Factory class for creating LLM providers."""

#     # Supported providers mapping
#     PROVIDERS = {
#         "ollama": ChatOllamaProvider,
#         "openai": ChatOpenAIProvider,
#         "groq": ChatGroqProvider,
#         "huggingface": ChatHuggingFaceProvider,
#     }

#     @classmethod
#     def create_provider(
#         cls,
#         provider_name: Optional[str] = None,
#         model_name: str = "HuggingFaceTB/SmolLM3-3B",
#         preload_model: bool = False,
#         **kwargs
#     ) -> LLMProvider:
#         """
#         Create an LLM provider instance.

#         Args:
#             provider_name: Name of the provider ("ollama", "openai", "groq", "huggingface")
#             model_name: Name of the model to use
#             **kwargs: Additional arguments for the provider

#         Returns:
#             LLMProvider instance

#         Raises:
#             ValueError: If provider is not supported
#         """
#         # Auto-detect provider from environment or use default
#         if not provider_name:
#             if os.getenv("OPENAI_API_KEY"):
#                 provider_name = "openai"
#             elif os.getenv("GROQ_API_KEY"):
#                 provider_name = "groq"
#             elif os.getenv("OLLAMA_BASE_URL"):
#                 provider_name = "ollama"
#             elif os.getenv("HUGGINGFACE_MODEL_PATH"):
#                 provider_name = "huggingface"
#             else:
#                 # Default to ollama if available, otherwise openai if key exists, fallback to huggingface
#                 provider_name = "ollama"  # Changed from huggingface to avoid server startup hanging

#         if provider_name not in cls.PROVIDERS:
#             raise ValueError(f"Unsupported provider: {provider_name}. Supported: {list(cls.PROVIDERS.keys())}")

#         provider_class = cls.PROVIDERS[provider_name]

#         # Filter out provider-specific parameters that shouldn't go to LangChain
#         # Only HuggingFace provider uses model_path and device parameters
#         if provider_name == 'huggingface':
#             filtered_kwargs = kwargs
#         else:
#             # For other providers, filter out HuggingFace-specific parameters
#             filtered_kwargs = {k: v for k, v in kwargs.items()
#                               if k not in ['preload_model', 'model_path', 'device']}

#         # Only pass preload_model to HuggingFace provider
#         if provider_name == 'huggingface':
#             return provider_class(
#                 model_name=model_name,
#                 preload_model=preload_model,
#                 **filtered_kwargs
#             )
#         else:
#             return provider_class(
#                 model_name=model_name,
#                 **filtered_kwargs
#             )

#     @classmethod
#     def get_available_providers(cls) -> Dict[str, bool]:
#         """Get availability status of all providers."""
#         status = {}
#         for name in cls.PROVIDERS.keys():
#             try:
#                 # For HuggingFace, just check if transformers is available without creating provider
#                 if name == "huggingface":
#                     try:
#                         import transformers
#                         status[name] = True
#                     except ImportError:
#                         status[name] = False
#                 else:
#                     provider = cls.create_provider(name)
#                     status[name] = provider.is_available()
#             except Exception:
#                 status[name] = False
#         return status

#     @classmethod
#     def list_supported_providers(cls) -> List[str]:
#         """List all supported provider names."""
#         return list(cls.PROVIDERS.keys())


# def create_llm_provider(
#     provider: Optional[str] = None,
#     model: Optional[str] = None,
#     preload_model: bool = False,
#     **kwargs
# ) -> LLMProvider:
#     """
#     Convenience function to create an LLM provider.

#     Args:
#         provider: Provider name ("ollama", "openai", "groq", "huggingface")
#         model: Model name
#         **kwargs: Additional provider-specific arguments

#     Returns:
#         LLMProvider instance
#     """
#     return LLMProviderFactory.create_provider(
#         provider_name=provider,
#         model_name=model or "HuggingFaceTB/SmolLM3-3B",
#         preload_model=preload_model,
#         **kwargs
#     )


from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import requests
import asyncio
import logging

from ..settings.settings import settings

logger = logging.getLogger(__name__)

try:
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage  # noqa: F401
except ImportError:
    # Fallback for type hints if langchain is not available
    BaseMessage = Any  # type: ignore
    HumanMessage = Any  # type: ignore
    SystemMessage = Any  # type: ignore
    AIMessage = Any  # type: ignore


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    async def generate_response(self, messages: List[Any]) -> str:
        """Generate a response from the given messages."""
        ...

    async def agenerate(self, messages: List[Any]) -> str:
        """Alias for generate_response for compatibility with pipeline nodes."""
        return await self.generate_response(messages)

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        ...

    def _extract_content(self, response: Any) -> str:
        """
        Extract content from various LLM response formats (LangChain messages,
        OpenAI-like dicts, etc.).
        """
        # LangChain messages / objects with .content
        if hasattr(response, "content") and response.content is not None:
            content = response.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list) and content:
                first_item = content[0]
                if isinstance(first_item, dict) and "text" in first_item and isinstance(first_item["text"], str):
                    return first_item["text"].strip()
                if isinstance(first_item, str):
                    return first_item.strip()

        # Dict-like responses
        if isinstance(response, dict):
            if "content" in response and isinstance(response["content"], str):
                return response["content"].strip()
            if "generated_text" in response and isinstance(response["generated_text"], str):
                return response["generated_text"].strip()

        # List from HF pipeline: [{"generated_text": "..."}]
        if isinstance(response, list) and response:
            first = response[0]
            if isinstance(first, dict) and "generated_text" in first:
                return str(first["generated_text"]).strip()

        # Fallback
        return str(response).strip()


class ChatOllamaProvider(LLMProvider):
    """Ollama LLM provider implementation."""

    def __init__(self, model_name: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url or (settings.model.ollama_base_url if getattr(settings, "model", None) else "http://127.0.0.1:11434")
        self._chat_model = None

    @property
    def provider_name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        """Check if Ollama service is running and model is available."""
        try:
            # Check server
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                print(f"⚠️  Ollama server not responding at {self.base_url}")
                return False

            # Check model presence
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
                model_temperature = settings.model.temperature if getattr(settings, "model", None) else 0.85
                self._chat_model = ChatOllama(
                    base_url=self.base_url,
                    model=self.model_name,
                    temperature=model_temperature,
                    **{k: v for k, v in self.kwargs.items() if k not in ["base_url"]}
                )
                print(f"✅ Ollama chat model created: {self.model_name}")
            except ImportError:
                raise RuntimeError("langchain_ollama not installed. Install with: pip install langchain_ollama")

        response = await self._chat_model.ainvoke(messages)
        return self._extract_content(response)


class ChatOpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self._chat_model = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.api_key)

    async def generate_response(self, messages: List[Any]) -> str:
        """Generate response using OpenAI."""
        if not self._chat_model:
            try:
                from langchain_openai import ChatOpenAI
                if not self.api_key:
                    raise RuntimeError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")

                model_temperature = settings.model.temperature if getattr(settings, "model", None) else 1.0
                self._chat_model = ChatOpenAI(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=model_temperature,
                    **{k: v for k, v in self.kwargs.items() if k not in ["api_key"]}
                )
            except ImportError:
                raise RuntimeError("langchain_openai not installed. Install with: pip install langchain_openai")

        response = await self._chat_model.ainvoke(messages)
        return self._extract_content(response)


class ChatGroqProvider(LLMProvider):
    """Groq LLM provider implementation."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()
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

                model_temperature = settings.model.temperature if getattr(settings, "model", None) else 1.0
                self._chat_model = ChatGroq(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=model_temperature,
                    **{k: v for k, v in self.kwargs.items() if k not in ["api_key"]}
                )
            except ImportError:
                raise RuntimeError("langchain_groq not installed. Install with: pip install langchain_groq")

        response = await self._chat_model.ainvoke(messages)
        return self._extract_content(response)


class ChatHuggingFaceProvider(LLMProvider):
    """HuggingFace local LLM provider implementation with chat template support."""

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device: str = "auto",
        preload_model: bool = False,
        max_new_tokens: Optional[int] = None,
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        default_model = "HuggingFaceTB/SmolLM3-3B" if not model_path else model_path
        self.model_path = model_path or default_model
        self.device = device
        self.preload_model = preload_model or os.getenv("HUGGINGFACE_PRELOAD_MODEL", "false").lower() == "true"
        self.max_new_tokens = max_new_tokens or getattr(settings.model, "max_tokens", None) or 32768
        self.timeout = timeout
        self._tokenizer = None
        self._model = None
        self._model_loading = False
        self._model_load_error = None

        if self.preload_model:
            try:
                self._load_model()
            except Exception as e:
                logger.warning(f"Failed to preload HuggingFace model {self.model_path}: {e}")
                self._model_load_error = str(e)

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def is_available(self) -> bool:
        """Check if HuggingFace transformers is installed."""
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self) -> None:
        """Load the HuggingFace model and tokenizer with error handling."""
        if self._model_loading:
            return

        self._model_loading = True
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            logger.info(f"Loading HuggingFace model: {self.model_path}")

            # Load tokenizer with timeout
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    timeout=self.timeout
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer for {self.model_path}: {e}")

            # Set pad token if not exists
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load model with timeout
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    dtype="auto",
                    device_map="auto" if torch.cuda.is_available() else None,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model {self.model_path}: {e}")

            logger.info(f"✅ Successfully loaded HuggingFace model: {self.model_path}")

        except Exception as e:
            self._model_load_error = str(e)
            logger.error(f"❌ Failed to load model {self.model_path}: {e}")
            raise
        finally:
            self._model_loading = False

    def _convert_messages_to_hf_format(self, messages: List[Any]) -> List[Dict[str, str]]:
        """Convert LangChain messages to HuggingFace chat format."""
        hf_messages = []
        
        for msg in messages:
            # Extract content from various message types
            if hasattr(msg, "content"):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get("content", str(msg))
            else:
                content = str(msg)
            
            # Skip empty messages
            if not content or not str(content).strip():
                continue
            
            # Determine role from message type
            # Check for LangChain message types first
            if hasattr(msg, "__class__"):
                class_name = msg.__class__.__name__
                # LangChain uses SystemMessage, HumanMessage, AIMessage
                if "System" in class_name or "system" in class_name.lower():
                    role = "system"
                elif "AI" in class_name or "Assistant" in class_name or "assistant" in class_name.lower() or "ai" in class_name.lower():
                    role = "assistant"
                elif "Human" in class_name or "User" in class_name or "human" in class_name.lower() or "user" in class_name.lower():
                    role = "user"
                else:
                    # Default based on position: first message is usually system
                    role = "system" if len(hf_messages) == 0 else "user"
            elif isinstance(msg, dict):
                # Dict format: check for 'role' key
                role = msg.get("role", "system" if len(hf_messages) == 0 else "user")
            else:
                # Fallback: first message is system, rest are user
                role = "system" if len(hf_messages) == 0 else "user"
            
            hf_messages.append({"role": role, "content": str(content)})
        
        return hf_messages

    async def generate_response(self, messages: List[Any]) -> str:
        """Generate response using HuggingFace local model with chat templates."""
        if self._model_load_error:
            raise RuntimeError(f"Model failed to load during initialization: {self._model_load_error}")

        # Load model if not already loaded
        if not self._model or not self._tokenizer:
            try:
                self._load_model()
            except Exception as e:
                raise RuntimeError(f"Failed to load HuggingFace model {self.model_path}: {e}")

        # Convert LangChain messages to HuggingFace chat format
        hf_messages = self._convert_messages_to_hf_format(messages)

        try:
            # Apply chat template if available, otherwise fall back to simple formatting
            if self._tokenizer.chat_template is not None:
                # Use chat template with generation prompt
                text = self._tokenizer.apply_chat_template(
                    hf_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Fallback: simple format
                logger.warning("No chat template found, using simple format")
                text_parts = []
                for msg in hf_messages:
                    if msg["role"] == "system":
                        text_parts.append(f"System: {msg['content']}")
                    elif msg["role"] == "user":
                        text_parts.append(f"User: {msg['content']}")
                    elif msg["role"] == "assistant":
                        text_parts.append(f"Assistant: {msg['content']}")
                text = "\n\n".join(text_parts) + "\n\nAssistant:"

            # Tokenize the input
            model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

            # Get model temperature from settings
            model_temperature = (
                settings.model.temperature 
                if getattr(settings, "model", None) and hasattr(settings.model, "temperature") 
                else 0.7
            )

            # Generate response in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _generate():
                import torch
                with torch.no_grad():
                    generated_ids = self._model.generate(
                        **model_inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=model_temperature,
                        do_sample=model_temperature > 0,
                        pad_token_id=self._tokenizer.eos_token_id,
                        **{k: v for k, v in self.kwargs.items() if k not in ["model_path", "device", "preload_model", "max_new_tokens", "timeout"]}
                    )
                return generated_ids

            # Generate with timeout
            generated_ids = await asyncio.wait_for(
                loop.run_in_executor(None, _generate),
                timeout=self.timeout,
            )

            # Extract only the new tokens (generated part)
            input_length = model_inputs.input_ids.shape[1]
            output_ids = generated_ids[0][input_length:]
            
            # Decode the generated tokens
            response = self._tokenizer.decode(output_ids, skip_special_tokens=True)
            
            return response.strip()

        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Model generation timed out for {self.model_path} after {self.timeout}s. "
                "The model may be overloaded or the request too complex."
            )
        except Exception as e:
            logger.error(f"Error generating response with HuggingFace model {self.model_path}: {e}")
            raise RuntimeError(f"Error generating response with HuggingFace model {self.model_path}: {e}")


class LLMProviderFactory:
    """Factory class for creating LLM providers."""

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
        **kwargs,
    ) -> LLMProvider:
        """
        Create an LLM provider instance.
        """
        # Auto-detect provider from environment if not given
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
                # Prefer Ollama if it's running locally; otherwise fall back to HF
                provider_name = "ollama"

        if provider_name not in cls.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider_name}. Supported: {list(cls.PROVIDERS.keys())}")

        provider_class = cls.PROVIDERS[provider_name]

        # Only HuggingFace uses preload_model/model_path/device explicitly
        if provider_name == "huggingface":
            filtered_kwargs = kwargs
            return provider_class(
                model_name=model_name,
                preload_model=preload_model,
                **filtered_kwargs,
            )

        # Filter out HF-only args for other providers
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["preload_model", "model_path", "device"]}
        return provider_class(
            model_name=model_name,
            **filtered_kwargs,
        )

    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """Get availability status of all providers."""
        status: Dict[str, bool] = {}
        for name in cls.PROVIDERS.keys():
            try:
                if name == "huggingface":
                    try:
                        import transformers  # noqa: F401
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
    **kwargs,
) -> LLMProvider:
    """Convenience function to create an LLM provider."""
    return LLMProviderFactory.create_provider(
        provider_name=provider,
        model_name=model or "HuggingFaceTB/SmolLM3-3B",
        preload_model=preload_model,
        **kwargs,
    )

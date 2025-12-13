from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Type, TYPE_CHECKING
import os
import requests
import asyncio
from ..settings.settings import settings
from pydantic import BaseModel
import logging

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

try:

    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
    from pydantic import BaseModel
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
    from pydantic import BaseModel
    # from langchain_community.llms import HuggingFacePipeline # Removed to avoid potential conflicts
except (ImportError, BaseException):
    # Fallback for type hints if langchain is not available or broken
    BaseMessage = Any
    HumanMessage = Any
    SystemMessage = Any
    AIMessage = Any
    BaseModel = Any


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    async def generate_response(self, messages: List[Any], structured_output: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None) -> Union[str, Any]:
        """Generate a response from the given messages.
        
        Args:
            messages: List of messages to send to the LLM
            structured_output: Optional Pydantic model or dict schema for structured output
            
        Returns:
            String response or structured object if structured_output is provided
        """
        pass
    
    async def agenerate(self, messages: List[Any], structured_output: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None) -> Union[str, Any]:
        """Alias for generate_response for compatibility with pipeline nodes."""
        return await self.generate_response(messages, structured_output)

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    def _extract_content(self, response: Any, structured_output: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None) -> Union[str, Any]:
        """Extract content from various LLM response formats."""
        # If structured output was requested and we got a structured response, return it directly
        if structured_output and isinstance(response, (dict, BaseModel)):
            return response
            
        if hasattr(response, 'content') and response.content is not None:
            content = response.content
            if isinstance(content, str):
                return content.strip()
            elif isinstance(content, list) and content:
                first_item = content
                if isinstance(first_item, dict) and 'text' in first_item and isinstance(first_item['text'], str):
                    return first_item['text'].strip()
                elif isinstance(first_item, str):
                    return first_item.strip()
        return str(response).strip() # Fallback to string conversion


class ChatOllamaProvider(LLMProvider):
    """Ollama LLM provider implementation."""

    def __init__(self, model_name: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url or (settings.model.ollama_base_url if settings.model else "http://127.0.0.1:11434")
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

    async def generate_response(self, messages: List[Any], structured_output: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None) -> Union[str, Any]:
        """Generate response using Ollama."""
        if not self._chat_model:
            try:
                from langchain_ollama import ChatOllama
                # Use the compatibility layer for gradual migration
                model_temperature = settings.model.temperature if settings.model else 0.85
                self._chat_model = ChatOllama(
                    model=self.model_name,
                    base_url=self.base_url,
                    temperature=model_temperature
                )
                print(f"✅ Ollama chat model created: {self.model_name}")
            except (ImportError, BaseException) as e:
                print(f"⚠️  Failed to initialize ChatOllama ({e}). Using HTTP fallback.")
                self._chat_model = None

        # Use structured output if requested
        if structured_output and self._chat_model:
            try:
                structured_model = self._chat_model.with_structured_output(structured_output)
                response = await structured_model.ainvoke(messages)
                return response
            except Exception as e:
                print(f"Warning: Structured output failed for Ollama, falling back to text: {e}")
                # Fall back to regular generation
        
        if self._chat_model:
            try:
                response = await self._chat_model.ainvoke(messages)
                content = self._extract_content(response)
            except Exception as e:
                print(f"❌ ChatOllama invocation failed: {e}. Trying HTTP fallback.")
                content = await self._generate_http(messages, structured_output)
        else:
            content = await self._generate_http(messages, structured_output)
        
        # If we need structured output but fell back to text, try to parse JSON manually
        if structured_output and isinstance(content, str):
            try:
                import json
                import re
                
                # Try to find JSON in the text
                json_str = content
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    json_str = match.group(0)
                
                parsed_json = json.loads(json_str)
                
                # If structured_output is a Pydantic model, instantiate it
                if isinstance(structured_output, type) and issubclass(structured_output, BaseModel):
                    return structured_output(**parsed_json)
                elif isinstance(structured_output, dict):
                    return parsed_json
                
            except Exception as parse_error:
                print(f"Warning: Manual JSON parsing failed: {parse_error}")
                
        return content

    async def _generate_http(self, messages: List[Any], structured_output: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None) -> str:
        """Generate response using direct HTTP request to Ollama."""
        import json
        
        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            role = "user"
            content = ""
            if hasattr(msg, "content"):
                content = msg.content
                if hasattr(msg, "type"):
                    if msg.type == "system": role = "system"
                    elif msg.type == "ai": role = "assistant"
            elif isinstance(msg, dict):
                content = msg.get("content", "")
                role = msg.get("role", "user")
            
            ollama_messages.append({"role": role, "content": content})
        
        # Add JSON formatting instruction if structured output is requested
        if structured_output:
            json_instruction = "\n\nIMPORTANT: Return your response ONLY as valid JSON with no additional text or formatting. Do not include explanations, headers, or any text before or after the JSON."
            if isinstance(structured_output, type) and issubclass(structured_output, BaseModel):
                # Get the fields from the Pydantic model to provide better instructions
                import inspect
                fields = inspect.signature(structured_output.__init__).parameters
                field_info = []
                for name, param in fields.items():
                    if name != 'self' and param.annotation:
                        field_type = param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)
                        field_info.append(f'"{name}": <{field_type}>')
                json_instruction += f"\n\nRequired fields: {', '.join(field_info)}"
            
            # Add the instruction to the last user message
            if ollama_messages:
                ollama_messages[-1]["content"] += json_instruction
            
        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": self.kwargs.get("temperature", 0.7)
            }
        }
        
        loop = asyncio.get_event_loop()
        
        def _do_request():
            try:
                resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=60)
                resp.raise_for_status()
                return resp.json().get("message", {}).get("content", "")
            except Exception as e:
                print(f"❌ Ollama HTTP request failed: {e}")
                return ""
                
        return await loop.run_in_executor(None, _do_request)
        
        # If we need structured output but fell back to text, try to parse JSON manually
        if structured_output and isinstance(content, str):
            try:
                import json
                import re
                
                # Try to find JSON in the text
                json_str = content
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    json_str = match.group(0)
                
                parsed_json = json.loads(json_str)
                
                # If structured_output is a Pydantic model, instantiate it
                if isinstance(structured_output, type) and issubclass(structured_output, BaseModel):
                    return structured_output(**parsed_json)
                elif isinstance(structured_output, dict):
                    return parsed_json
                
            except Exception as parse_error:
                print(f"Warning: Manual JSON parsing failed: {parse_error}")
                
        return content


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

    async def generate_response(self, messages: List[Any], structured_output: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None) -> Union[str, Any]:
        """Generate response using OpenAI."""
        if not self._chat_model:
            try:
                from langchain_openai import ChatOpenAI
                if not self.api_key:
                    raise RuntimeError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")

                # Use the compatibility layer for gradual migration
                model_temperature = settings.model.temperature if settings.model else 1.0
                self._chat_model = ChatOpenAI(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=model_temperature,
                    **{k: v for k, v in self.kwargs.items() if k not in ['api_key']}
                )
            except ImportError:
                raise RuntimeError("langchain_openai not installed. Install with: pip install langchain_openai")

        # Use structured output if requested
        if structured_output:
            try:
                structured_model = self._chat_model.with_structured_output(structured_output)
                response = await structured_model.ainvoke(messages)
                return response
            except Exception as e:
                print(f"Warning: Structured output failed for OpenAI, falling back to text: {e}")
                # Fall back to regular generation

        response = await self._chat_model.ainvoke(messages)
        return self._extract_content(response, structured_output)


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

    async def generate_response(self, messages: List[Any], structured_output: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None) -> Union[str, Any]:
        """Generate response using Groq."""
        if not self._chat_model:
            try:
                from langchain_groq import ChatGroq
                if not self.api_key:
                    raise RuntimeError("Groq API key not configured. Set GROQ_API_KEY environment variable.")

                # Use the compatibility layer for gradual migration
                model_temperature = settings.model.temperature if settings.model else 1.0
                self._chat_model = ChatGroq(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=model_temperature,
                    **{k: v for k, v in self.kwargs.items() if k not in ['api_key']}
                )
            except ImportError:
                raise RuntimeError("langchain_groq not installed. Install with: pip install langchain_groq")

        # Use structured output if requested
        if structured_output:
            try:
                structured_model = self._chat_model.with_structured_output(structured_output)
                response = await structured_model.ainvoke(messages)
                return response
            except Exception as e:
                print(f"Warning: Structured output failed for Groq, falling back to text: {e}")
                # Fall back to regular generation

        response = await self._chat_model.ainvoke(messages)
        return self._extract_content(response, structured_output)


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

    async def generate_response(self, messages: List[Any], structured_output: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None) -> Union[str, Any]:
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
            response_text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
            response_text = response_text.strip()
            
            # Handle structured output if requested
            if structured_output:
                try:
                    import json
                    parsed_json = None
                    
                    # Try to extract JSON from the response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        try:
                            parsed_json = json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
                    
                    # If extraction failed, try parsing the whole response
                    if parsed_json is None:
                        try:
                            parsed_json = json.loads(response_text)
                        except json.JSONDecodeError:
                            pass
                    
                    # If we successfully parsed JSON, format it according to structured_output type
                    if parsed_json is not None:
                        # If structured_output is a Pydantic model, try to instantiate it
                        if isinstance(structured_output, type) and issubclass(structured_output, BaseModel):
                            try:
                                return structured_output(**parsed_json)
                            except (TypeError, ValueError) as e:
                                logger.warning(f"Failed to instantiate Pydantic model from JSON: {e}")
                                return parsed_json
                        # If it's a dict schema, return the parsed JSON
                        elif isinstance(structured_output, dict):
                            return parsed_json
                        else:
                            return parsed_json
                    else:
                        logger.warning(f"No valid JSON found in HuggingFace response for structured output, falling back to text")
                        return response_text
                except Exception as e:
                    logger.warning(f"Failed to parse structured output from HuggingFace response, falling back to text: {e}")
                    # Fall back to text response
                    return response_text
            
            return response_text

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
# Example usage with simplified structured output
"""
Example usage with structured output passed directly to generate_response:

from pydantic import BaseModel
from typing import List

class EmotionAnalysis(BaseModel):
    emotion: str
    confidence: float
    reasoning: str

# Create provider
provider = create_llm_provider("openai", "gpt-4")

# Use structured output directly in generate_response
messages = [
    SystemMessage(content="You are an emotion analysis expert."),
    HumanMessage(content="I'm feeling really anxious about my job interview tomorrow.")
]

# Pass structured output schema directly to the method
response = await provider.generate_response(messages, structured_output=EmotionAnalysis)
# response will be an EmotionAnalysis object with emotion, confidence, and reasoning fields

# Or with dictionary schema:
schema = {
    "type": "object",
    "properties": {
        "emotion": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"}
    },
    "required": ["emotion", "confidence", "reasoning"]
}

response = await provider.generate_response(messages, structured_output=schema)
# response will be a dictionary matching the schema

# Regular text response (no structured output)
text_response = await provider.generate_response(messages)
# text_response will be a string
"""
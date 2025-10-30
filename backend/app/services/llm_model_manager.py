import requests
from typing import Optional
from langchain_ollama import ChatOllama
# Use the compatibility layer for gradual migration
from ..settings.settings import settings
from ..prompts.system_prompts import SystemPrompts


class ModelManager:
    """Manages LLM model initialization and health checks."""

    def __init__(self, model_name: Optional[str] = None, use_vllm: bool = False):
        self.model_name = model_name or (settings.model.model_name if settings.model else "gemma3:1b")
        self.use_vllm = use_vllm
        self.ollama_base_url = settings.model.ollama_base_url if settings.model else "http://127.0.0.1:11434"
        self.vllm_base_url = settings.model.vllm_base_url if settings.model else "http://127.0.0.1:8001/v1"
        self.chat_model = None
        self.is_initialized = False

    def initialize_model(self) -> bool:
        """Initialize the appropriate model based on configuration"""
        try:
            if self.use_vllm:
                return self._initialize_vllm_model()
            else:
                return self._initialize_ollama_model()
        except Exception as e:
            print(f"Error during model initialization: {e}")
            return False

    def _initialize_vllm_model(self) -> bool:
        """Initialize vLLM model"""
        if not self._is_vllm_running():
            print(SystemPrompts.get_error_messages()["vllm_not_running"])
            return False

        try:
            self.chat_model = ChatOllama(
                base_url=self.vllm_base_url,
                model=self.model_name,
                temperature=settings.model.temperature if settings.model else 0.85
            )
            self.is_initialized = True
            print("LLM initialized with vLLM (Llama-3-8B-Instruct)")
            return True
        except Exception as e:
            print(f"vLLM initialization error: {e}")
            return False

    def _initialize_ollama_model(self) -> bool:
        """Initialize Ollama model"""
        if not self._is_ollama_running():
            print(SystemPrompts.get_error_messages()["ollama_not_running"])
            return False

        if not self._is_model_available():
            print(SystemPrompts.get_error_messages()["model_not_available"].format(model_name=self.model_name))
            return False

        try:
            self.chat_model = ChatOllama(
                base_url=self.ollama_base_url,
                model=self.model_name,
                temperature=settings.model.temperature if settings.model else 0.85
            )
            self.is_initialized = True
            print(f"LLM initialized with Ollama model: {self.model_name}")
            return True
        except Exception as e:
            print(f"Ollama initialization error: {e}")
            return False

    def _is_ollama_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def _is_model_available(self) -> bool:
        """Check if the specified model is available in Ollama"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            names = [m.get("name", "") for m in models]
            print(f"[Ollama] Available models: {names}")
            # Treat exact or prefix matches (llama3 matches llama3.1)
            return any(self.model_name == n or n.startswith(self.model_name) for n in names)
        except Exception:
            return False

    def _is_vllm_running(self) -> bool:
        """Check if vLLM service is running"""
        try:
            response = requests.get(f"{self.vllm_base_url}/models")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    async def generate_response(self, messages) -> str:
        """Generate response using the initialized model"""
        if not self.is_initialized or not self.chat_model:
            return SystemPrompts.get_error_messages()["model_not_initialized"]

        response = await self.chat_model.ainvoke(messages)
        content = response.content
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, list) and content:
            # Assuming it's a list of dicts with 'text' key, or just strings
            first_item = content
            if isinstance(first_item, dict) and 'text' in first_item:
                text_content = first_item.get('text')
                if isinstance(text_content, str):
                    return text_content.strip()
            elif isinstance(first_item, str):
                return first_item.strip()
        return str(content).strip() # Fallback to string conversion
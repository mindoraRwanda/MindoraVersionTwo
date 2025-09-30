"""
Model management and initialization for LLM service.
"""
import requests
from typing import Optional
from langchain_ollama import ChatOllama
from .llm_config import (
    DEFAULT_MODEL_NAME, OLLAMA_BASE_URL, VLLM_BASE_URL,
    ERROR_MESSAGES
)


class ModelManager:
    """Manages LLM model initialization and health checks."""

    def __init__(self, model_name: str = None, use_vllm: bool = False):
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.use_vllm = use_vllm
        self.ollama_base_url = OLLAMA_BASE_URL
        self.vllm_base_url = VLLM_BASE_URL
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
            print(ERROR_MESSAGES["vllm_not_running"])
            return False

        try:
            self.chat_model = ChatOllama(
                base_url=self.vllm_base_url,
                model=self.model_name,
                temperature=0.85
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
            print(ERROR_MESSAGES["ollama_not_running"])
            return False

        if not self._is_model_available():
            print(ERROR_MESSAGES["model_not_available"].format(model_name=self.model_name))
            return False

        try:
            self.chat_model = ChatOllama(
                base_url=self.ollama_base_url,
                model=self.model_name,
                temperature=0.85
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
            return ERROR_MESSAGES["model_not_initialized"]

        response = await self.chat_model.ainvoke(messages)
        return response.content.strip()
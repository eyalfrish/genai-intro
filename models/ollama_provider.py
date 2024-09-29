from typing import Any

from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from models.model_provider import ModelProvider


class OllamaModelProvider(ModelProvider):
    @staticmethod
    def validate_provider() -> tuple[bool, str]:
        if not OllamaModelProvider._is_ollama_running():
            return False, "Ollama server is not running"
        return True, "Validation successful"

    @staticmethod
    def _get_llm_instance(model_name: str, **kwargs: Any) -> BaseChatModel:
        return ChatOllama(model=model_name, **kwargs)

    @staticmethod
    def _get_embedder_instance(model_name: str, **kwargs: Any) -> Embeddings:
        return OllamaEmbeddings(model=model_name, **kwargs)

    @staticmethod
    def _get_vision_instance(model_name: str, **kwargs: Any) -> BaseChatModel:
        return ChatOllama(model=model_name, **kwargs)

    @staticmethod
    def _get_default_llm_model_name() -> str:
        return "llama3.1:8b"

    @staticmethod
    def _get_default_embedder_model_name() -> str:
        return "mxbai-embed-large:latest"

    @staticmethod
    def _get_default_vision_model_name() -> str:
        return "llava-phi3:3.8b"

    @staticmethod
    def _is_ollama_running() -> bool:
        import httpx
        import ollama

        try:
            ollama.list()
            return True
        except httpx.ConnectError:
            return False

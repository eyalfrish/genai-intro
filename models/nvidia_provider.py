import os
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from models.model_provider import ModelProvider


class NvidiaModelProvider(ModelProvider):
    @staticmethod
    def validate_provider() -> tuple[bool, str]:
        if not os.getenv("NVIDIA_API_KEY"):
            return False, "NVIDIA_API_KEY is not defined"
        return True, "Validation successful"

    @staticmethod
    def _get_llm_instance(model_name: str, **kwargs: Any) -> BaseChatModel:
        return ChatNVIDIA(model=model_name, **kwargs)

    @staticmethod
    def _get_embedder_instance(model_name: str, **kwargs: Any) -> Embeddings:
        return NVIDIAEmbeddings(model=model_name, **kwargs)

    @staticmethod
    def _get_vision_instance(model_name: str, **kwargs: Any) -> BaseChatModel:
        return ChatNVIDIA(
            model=model_name,
            **kwargs,
        )

    @staticmethod
    def _get_default_llm_model_name() -> str:
        return "meta/llama-3.3-70b-instruct"

    @staticmethod
    def _get_default_embedder_model_name() -> str:
        return "NV-Embed-QA"

    @staticmethod
    def _get_default_vision_model_name() -> str:
        return "meta/llama-3.2-90b-vision-instruct"

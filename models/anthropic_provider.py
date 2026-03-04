import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_voyageai import VoyageAIEmbeddings

from models.model_provider import ModelProvider


class AnthropicModelProvider(ModelProvider):
    @staticmethod
    def validate_provider() -> tuple[bool, str]:
        if not os.getenv("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY is not defined"
        return True, "Validation successful"

    @staticmethod
    def _get_llm_instance(model_name: str, **kwargs: Any) -> BaseChatModel:
        return ChatAnthropic(model=model_name, **kwargs)  # type: ignore[unknown-argument]  # ty doesn't resolve Pydantic __init__ fields

    @staticmethod
    def _get_embedder_instance(model_name: str, **kwargs: Any) -> Embeddings:
        return VoyageAIEmbeddings(model=model_name, batch_size=128, **kwargs)

    @staticmethod
    def _get_vision_instance(model_name: str, **kwargs: Any) -> BaseChatModel:
        return ChatAnthropic(model=model_name, **kwargs)  # type: ignore[unknown-argument]  # ty doesn't resolve Pydantic __init__ fields

    @staticmethod
    def _get_default_llm_model_name() -> str:
        return "claude-3-5-sonnet-20241022"

    @staticmethod
    def _get_default_embedder_model_name() -> str:
        return "voyage-3-lite"

    @staticmethod
    def _get_default_vision_model_name() -> str:
        return "claude-3-5-sonnet-20241022"

import os
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from models.model_provider import ModelProvider


class OpenAIModelProvider(ModelProvider):
    @staticmethod
    def validate_provider() -> tuple[bool, str]:
        if not os.getenv("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY is not defined"
        return True, "Validation successful"

    @staticmethod
    def _get_llm_instance(model_name: str, **kwargs: Any) -> BaseChatModel:
        return ChatOpenAI(model_name=model_name, **kwargs)  # type: ignore[call-arg]

    @staticmethod
    def _get_embedder_instance(model_name: str, **kwargs: Any) -> Embeddings:
        embedder = OpenAIEmbeddings(model=model_name, **kwargs)
        assert issubclass(type(embedder), Embeddings)
        return embedder

    @staticmethod
    def _get_vision_instance(model_name: str, **kwargs: Any) -> BaseChatModel:
        return ChatOpenAI(model_name=model_name, **kwargs)  # type: ignore[call-arg]

    @staticmethod
    def _get_default_llm_model_name() -> str:
        return "gpt-4o-mini"

    @staticmethod
    def _get_default_embedder_model_name() -> str:
        return "text-embedding-3-small"

    @staticmethod
    def _get_default_vision_model_name() -> str:
        return "gpt-4o-mini"

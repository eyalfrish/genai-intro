from abc import ABC, abstractmethod
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel


class ModelProvider(ABC):
    @staticmethod
    def initialize_provider() -> None:
        pass

    @classmethod
    def get_llm_instance(
        cls, external_model_name: str | None = None, **kwargs: Any
    ) -> BaseChatModel:
        return cls._get_llm_instance(
            external_model_name or cls._get_default_llm_model_name(), **kwargs
        )

    @classmethod
    def get_embedder_instance(
        cls, external_model_name: str | None = None, **kwargs: Any
    ) -> Embeddings:
        return cls._get_embedder_instance(
            external_model_name or cls._get_default_embedder_model_name(), **kwargs
        )

    @classmethod
    def get_vision_instance(
        cls, external_model_name: str | None = None, **kwargs: Any
    ) -> BaseChatModel:
        return cls._get_vision_instance(
            external_model_name or cls._get_default_vision_model_name(), **kwargs
        )

    @staticmethod
    def supports_structured_output() -> bool:
        return True

    @staticmethod
    @abstractmethod
    def validate_provider() -> tuple[bool, str]:
        pass

    @staticmethod
    @abstractmethod
    def _get_llm_instance(model_name: str, **kwargs: Any) -> BaseChatModel:
        pass

    @staticmethod
    @abstractmethod
    def _get_embedder_instance(model_name: str, **kwargs: Any) -> Embeddings:
        pass

    @staticmethod
    @abstractmethod
    def _get_vision_instance(model_name: str, **kwargs: Any) -> BaseChatModel:
        pass

    @staticmethod
    @abstractmethod
    def _get_default_llm_model_name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def _get_default_embedder_model_name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def _get_default_vision_model_name() -> str:
        pass

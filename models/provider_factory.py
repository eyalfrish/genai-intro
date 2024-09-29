from argparse import ArgumentParser, Namespace

from models.anthropic_provider import AnthropicModelProvider
from models.model_provider import ModelProvider
from models.nvidia_provider import NvidiaModelProvider
from models.ollama_provider import OllamaModelProvider
from models.openai_provider import OpenAIModelProvider


class ProviderFactory:
    PROVIDERS_DICT: dict[str, type[ModelProvider]] = {
        "nvidia": NvidiaModelProvider,
        "openai": OpenAIModelProvider,
        "ollama": OllamaModelProvider,
        "anthropic": AnthropicModelProvider,
    }

    @staticmethod
    def get_provider(provider_name: str) -> ModelProvider:
        if provider_name in ProviderFactory.PROVIDERS_DICT:
            provider_class = ProviderFactory.PROVIDERS_DICT[provider_name]
            is_valid, message = provider_class.validate_provider()
            if not is_valid:
                raise ValueError(f"Provider validation failed: {message}")
            return provider_class()
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    @staticmethod
    def list_providers() -> list[str]:
        return list(ProviderFactory.PROVIDERS_DICT.keys())

    @staticmethod
    def add_provider_arg(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--inference-provider",
            choices=ProviderFactory.list_providers(),
            default="nvidia",
            help="The inference provider to use.",
        )
        return parser

    @staticmethod
    def parse_provider_arg() -> Namespace:
        parser = ArgumentParser(description="Inference provider argument parser")
        return ProviderFactory.add_provider_arg(parser).parse_args()

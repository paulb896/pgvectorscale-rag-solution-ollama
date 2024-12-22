from typing import Any, Dict, List, Type

# import instructor
from services.ollama_client import OllamaClient
from pydantic import BaseModel

from config.settings import get_settings

class LLMFactory:
    def __init__(self, provider: str):
        if provider != "ollama":
            raise ValueError("Currently, only 'ollama' is supported for local use.")
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        return OllamaClient(base_url=self.settings.base_url)

    def create_completion(
        self, messages: List[Dict[str, str]]
    ) -> Any:
        return self.client.chat(
            model=self.settings.default_model,
            messages=messages,
            temperature=self.settings.temperature,
        )
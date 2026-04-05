"""Custom endpoint provider — any OpenAI-compatible API.

Works with: Ollama, LM Studio, vLLM, any self-hosted endpoint,
or paid APIs (OpenAI, Anthropic, Groq, Together, etc.)

The user provides a base_url and optional api_key in config.yaml:

    provider:
      custom:
        base_url: http://localhost:11434/v1
        api_key: null
        model: llama3
"""

from __future__ import annotations

from typing import Iterator

from .base import Provider, ProviderModel, ProviderStatus
from ..core.config import get_config


class CustomProvider(Provider):
    name = "custom"
    display_name = "Custom Endpoint"
    install_command = None  # User manages their own endpoint

    def _get_config(self) -> tuple[str | None, str | None, str | None]:
        cfg = get_config()
        return (
            cfg.get("provider.custom.base_url"),
            cfg.get("provider.custom.api_key"),
            cfg.get("provider.custom.model"),
        )

    def detect(self) -> bool:
        base_url, _, _ = self._get_config()
        return base_url is not None and base_url != ""

    def status(self) -> ProviderStatus:
        base_url, api_key, model = self._get_config()
        if not base_url:
            return ProviderStatus(installed=False, error="No base_url configured")

        # Don't hit the API on status check — just report configured model
        models = [ProviderModel(id=model, name=model)] if model else []
        return ProviderStatus(
            installed=True,
            authenticated=bool(api_key),
            version=base_url,
            models=models,
        )

    def _get_client(self):
        from openai import OpenAI
        base_url, api_key, _ = self._get_config()
        if not base_url:
            raise RuntimeError("Custom endpoint not configured. Set provider.custom.base_url in config.yaml")
        return OpenAI(base_url=base_url, api_key=api_key or "none", max_retries=0)

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        client = self._get_client()
        _, _, default_model = self._get_config()
        model = model or default_model or "default"
        print(f"  [custom] chat model={model}")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1500,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  [custom] chat ERROR: {e}")
            raise

    def stream(self, messages: list[dict], model: str | None = None) -> Iterator[str]:
        client = self._get_client()
        _, _, default_model = self._get_config()
        model = model or default_model or "default"
        print(f"  [custom] stream model={model}")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1500,
                stream=True,
            )
            for chunk in response:
                text = chunk.choices[0].delta.content or ""
                if text:
                    yield text
        except Exception as e:
            print(f"  [custom] stream ERROR: {e}")
            raise

"""Provider registry — manages the active LLM provider."""

from __future__ import annotations

from .base import Provider, ProviderStatus
from ..core.config import get_config


class ProviderRegistry:
    """Manages available LLM providers."""

    def __init__(self):
        self._providers: dict[str, Provider] = {}
        self._active: str | None = None
        self._register_all()

    def _register_all(self):
        from .custom import CustomProvider

        provider = CustomProvider()
        self._providers[provider.name] = provider

        cfg = get_config()
        self._active = cfg.get("provider.active", "custom")

    @property
    def active(self) -> Provider | None:
        return self._providers.get(self._active)

    @active.setter
    def active(self, name: str):
        if name not in self._providers:
            raise ValueError(f"Unknown provider: {name}. Available: {list(self._providers.keys())}")
        self._active = name

    def get(self, name: str) -> Provider | None:
        return self._providers.get(name)

    def all_status(self) -> dict[str, dict]:
        result = {}
        for name, provider in self._providers.items():
            try:
                status = provider.status()
            except Exception as e:
                status = ProviderStatus(error=str(e))

            result[name] = {
                "name": name,
                "display_name": provider.display_name,
                "installed": status.installed,
                "authenticated": status.authenticated,
                "version": status.version,
                "error": status.error,
                "models": [
                    {"id": m.id, "name": m.name, "free": m.free, "context_window": m.context_window}
                    for m in status.models
                ],
                "free_model_count": sum(1 for m in status.models if m.free),
                "is_active": name == self._active,
            }
        return result


_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry

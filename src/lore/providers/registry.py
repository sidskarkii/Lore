"""Provider registry — detect, manage, and switch between LLM providers.

The registry scans for installed CLI tools on startup and exposes them
to the API layer. It manages the active provider and handles install requests.
"""

from __future__ import annotations

from .base import Provider, ProviderStatus
from ..core.config import get_config


class ProviderRegistry:
    """Manages all available LLM providers."""

    def __init__(self):
        self._providers: dict[str, Provider] = {}
        self._active: str | None = None
        self._register_all()

    def _register_all(self):
        """Import and register all provider implementations."""
        from .claude_code import ClaudeCodeProvider
        from .opencode import OpenCodeProvider
        from .codex import CodexProvider
        from .kilo import KiloProvider
        from .custom import CustomProvider

        for cls in [ClaudeCodeProvider, OpenCodeProvider, CodexProvider, KiloProvider, CustomProvider]:
            provider = cls()
            self._providers[provider.name] = provider

        # Set active from config
        cfg = get_config()
        self._active = cfg.get("provider.active", "kilo")

    @property
    def active(self) -> Provider | None:
        """The currently active provider."""
        return self._providers.get(self._active)

    @active.setter
    def active(self, name: str):
        if name not in self._providers:
            raise ValueError(f"Unknown provider: {name}. Available: {list(self._providers.keys())}")
        self._active = name

    def get(self, name: str) -> Provider | None:
        """Get a provider by name."""
        return self._providers.get(name)

    def all_status(self) -> dict[str, dict]:
        """Get status of every registered provider.

        Returns a dict keyed by provider name with status info + metadata.
        """
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
                "user": status.user,
                "error": status.error,
                "models": [
                    {"id": m.id, "name": m.name, "free": m.free, "context_window": m.context_window}
                    for m in status.models
                ],
                "free_model_count": sum(1 for m in status.models if m.free),
                "install_command": provider.install_command,
                "is_active": name == self._active,
            }
        return result

    def detect_all(self) -> dict[str, bool]:
        """Quick check: which providers are installed?"""
        return {name: p.detect() for name, p in self._providers.items()}

    def install(self, name: str) -> bool:
        """Install a provider's CLI tool. Returns True on success."""
        provider = self._providers.get(name)
        if provider is None:
            return False
        return provider.install()

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        """Chat using the active provider."""
        provider = self.active
        if provider is None:
            raise RuntimeError("No active provider. Configure one in settings.")
        if not provider.detect():
            raise RuntimeError(f"Provider '{self._active}' is not installed.")
        return provider.chat(messages, model)

    def stream(self, messages: list[dict], model: str | None = None):
        """Stream chat using the active provider."""
        provider = self.active
        if provider is None:
            raise RuntimeError("No active provider. Configure one in settings.")
        if not provider.detect():
            raise RuntimeError(f"Provider '{self._active}' is not installed.")
        yield from provider.stream(messages, model)


# Module-level singleton
_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Get or create the global provider registry."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry

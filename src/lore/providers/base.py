"""Provider base — abstract interface for LLM inference backends.

Every provider (CLI tool, local model, API endpoint) implements this
interface. The API layer and UI don't care which provider is active —
they just call chat() and stream().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class ProviderModel:
    """A model available through a provider."""
    id: str
    name: str
    free: bool = False
    context_window: int = 0


@dataclass
class ProviderStatus:
    """Current state of a provider."""
    installed: bool = False
    authenticated: bool = False
    version: str | None = None
    user: str | None = None
    error: str | None = None
    models: list[ProviderModel] = field(default_factory=list)


class Provider(ABC):
    """Abstract base for all LLM providers."""

    # Subclasses must set these
    name: str = ""
    display_name: str = ""
    install_command: str | None = None  # e.g. "npm install -g @kilocode/cli"

    @abstractmethod
    def detect(self) -> bool:
        """Check if this provider's CLI/tool is installed on the system."""

    @abstractmethod
    def status(self) -> ProviderStatus:
        """Get detailed status: installed, auth, version, available models."""

    @abstractmethod
    def chat(self, messages: list[dict], model: str | None = None) -> str:
        """Send messages and get a complete response.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": str}
            model: Model ID override (uses config default if None)

        Returns:
            The assistant's response text.
        """

    @abstractmethod
    def stream(self, messages: list[dict], model: str | None = None) -> Iterator[str]:
        """Send messages and yield response chunks as they arrive.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": str}
            model: Model ID override (uses config default if None)

        Yields:
            Text chunks of the response.
        """

    def free_models(self) -> list[ProviderModel]:
        """Return models that cost nothing to use."""
        st = self.status()
        return [m for m in st.models if m.free]

    def install(self) -> bool:
        """Attempt to install this provider's CLI. Returns True on success."""
        if not self.install_command:
            return False
        import subprocess
        try:
            result = subprocess.run(
                self.install_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            return result.returncode == 0
        except Exception:
            return False

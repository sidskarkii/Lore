"""Provider base — abstract interface for LLM inference backends."""

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
    error: str | None = None
    models: list[ProviderModel] = field(default_factory=list)


class Provider(ABC):
    """Abstract base for all LLM providers."""

    name: str = ""
    display_name: str = ""

    @abstractmethod
    def detect(self) -> bool:
        """Check if this provider is configured and available."""

    @abstractmethod
    def status(self) -> ProviderStatus:
        """Get detailed status: installed, auth, available models."""

    @abstractmethod
    def chat(self, messages: list[dict], model: str | None = None) -> str:
        """Send messages and get a complete response."""

    @abstractmethod
    def stream(self, messages: list[dict], model: str | None = None) -> Iterator[str]:
        """Send messages and yield response chunks as they arrive."""

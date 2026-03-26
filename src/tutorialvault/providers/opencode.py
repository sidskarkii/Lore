"""OpenCode provider — uses the `opencode` CLI with 6 free models.

Invocation:
    opencode run "{prompt}" -m opencode/minimax-m2.5-free --format json

Auth: OAuth for Anthropic/OpenAI, or use free opencode/* models with no auth.
Check: `opencode auth list`
Free models: opencode/mimo-v2-omni-free, opencode/minimax-m2.5-free, etc.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Iterator

from .base import Provider, ProviderModel, ProviderStatus
from ..core.config import get_config

# Known free models (verified on this machine)
_FREE_MODELS = [
    ProviderModel(id="opencode/minimax-m2.5-free", name="MiniMax M2.5 (free)", free=True, context_window=204_800),
    ProviderModel(id="opencode/mimo-v2-omni-free", name="MiMo V2 Omni (free)", free=True, context_window=262_144),
    ProviderModel(id="opencode/mimo-v2-pro-free", name="MiMo V2 Pro (free)", free=True, context_window=1_048_576),
    ProviderModel(id="opencode/nemotron-3-super-free", name="Nemotron 3 Super (free)", free=True, context_window=262_144),
    ProviderModel(id="opencode/gpt-5-nano", name="GPT-5 Nano", free=True, context_window=128_000),
    ProviderModel(id="opencode/big-pickle", name="Big Pickle", free=True, context_window=128_000),
]


class OpenCodeProvider(Provider):
    name = "opencode"
    display_name = "OpenCode"
    install_command = "npm install -g opencode-ai"

    def _bin(self) -> str | None:
        return shutil.which("opencode")

    def detect(self) -> bool:
        return self._bin() is not None

    def status(self) -> ProviderStatus:
        binary = self._bin()
        if not binary:
            return ProviderStatus(installed=False)

        version = None
        try:
            r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10)
            version = r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            pass

        # Check auth
        authenticated = False
        user = None
        try:
            r = subprocess.run(
                [binary, "auth", "list"],
                capture_output=True, text=True, timeout=10,
            )
            output = r.stdout
            # opencode auth list shows credentials with bullet points
            authenticated = "credential" in output.lower() or "oauth" in output.lower()
        except Exception:
            pass

        return ProviderStatus(
            installed=True,
            authenticated=authenticated,
            version=version,
            user=user,
            models=list(_FREE_MODELS),
        )

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        binary = self._bin()
        if not binary:
            raise RuntimeError("OpenCode CLI not found")

        cfg = get_config()
        model = model or cfg.get("provider.opencode.model", "opencode/minimax-m2.5-free")

        # Flatten messages into a single prompt
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

        result = subprocess.run(
            [binary, "run", prompt, "-m", model, "--format", "json"],
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(f"OpenCode error: {result.stderr[:500]}")

        # Parse JSON output — opencode emits JSONL events
        text_parts = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                # Look for text content in various event formats
                if isinstance(event, dict):
                    content = event.get("content", event.get("text", ""))
                    if content:
                        text_parts.append(content)
            except json.JSONDecodeError:
                text_parts.append(line)

        return "\n".join(text_parts) if text_parts else result.stdout.strip()

    def stream(self, messages: list[dict], model: str | None = None) -> Iterator[str]:
        binary = self._bin()
        if not binary:
            raise RuntimeError("OpenCode CLI not found")

        cfg = get_config()
        model = model or cfg.get("provider.opencode.model", "opencode/minimax-m2.5-free")
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

        proc = subprocess.Popen(
            [binary, "run", prompt, "-m", model, "--format", "json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    content = event.get("content", event.get("text", ""))
                    if content:
                        yield content
            except json.JSONDecodeError:
                yield line

        proc.wait()

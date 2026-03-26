"""Codex CLI provider — uses OpenAI's `codex` CLI with existing subscription.

Invocation:
    codex exec "{prompt}" -m o4-mini --json --full-auto --ephemeral

Auth: OAuth via `codex login` or OPENAI_API_KEY env var.
Check: `codex login status`
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Iterator

from .base import Provider, ProviderModel, ProviderStatus
from ..core.config import get_config


class CodexProvider(Provider):
    name = "codex"
    display_name = "Codex CLI"
    install_command = "npm install -g @openai/codex"

    def _bin(self) -> str | None:
        return shutil.which("codex")

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

        authenticated = False
        try:
            r = subprocess.run(
                [binary, "login", "status"],
                capture_output=True, text=True, timeout=10,
            )
            authenticated = r.returncode == 0
        except Exception:
            pass

        models = [
            ProviderModel(id="o4-mini", name="O4 Mini", context_window=128_000),
            ProviderModel(id="o3", name="O3", context_window=200_000),
            ProviderModel(id="gpt-4.1", name="GPT-4.1", context_window=1_000_000),
        ]

        return ProviderStatus(
            installed=True,
            authenticated=authenticated,
            version=version,
            models=models,
        )

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        binary = self._bin()
        if not binary:
            raise RuntimeError("Codex CLI not found")

        cfg = get_config()
        model = model or cfg.get("provider.codex.model", "o4-mini")
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

        result = subprocess.run(
            [
                binary, "exec", prompt,
                "-m", model,
                "--json",
                "--full-auto",
                "--ephemeral",
            ],
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Codex error: {result.stderr[:500]}")

        # Parse JSONL output
        text_parts = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    content = event.get("content", event.get("text", event.get("message", "")))
                    if content:
                        text_parts.append(content)
            except json.JSONDecodeError:
                text_parts.append(line)

        return "\n".join(text_parts) if text_parts else result.stdout.strip()

    def stream(self, messages: list[dict], model: str | None = None) -> Iterator[str]:
        binary = self._bin()
        if not binary:
            raise RuntimeError("Codex CLI not found")

        cfg = get_config()
        model = model or cfg.get("provider.codex.model", "o4-mini")
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

        proc = subprocess.Popen(
            [
                binary, "exec", prompt,
                "-m", model,
                "--json",
                "--full-auto",
                "--ephemeral",
            ],
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

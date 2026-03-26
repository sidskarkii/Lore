"""Claude Code provider — uses the `claude` CLI with the user's existing subscription.

Invocation:
    claude -p "{prompt}" --bare --output-format json --model sonnet \
        --max-turns 1 --no-session-persistence

Auth: OAuth via Claude Max/Pro/Teams subscription.
Check: `claude auth status` (exit code 0 = logged in).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Iterator

from .base import Provider, ProviderModel, ProviderStatus
from ..core.config import get_config


class ClaudeCodeProvider(Provider):
    name = "claude_code"
    display_name = "Claude Code"
    install_command = "npm install -g @anthropic-ai/claude-code"

    def _bin(self) -> str | None:
        return shutil.which("claude")

    def detect(self) -> bool:
        return self._bin() is not None

    def status(self) -> ProviderStatus:
        binary = self._bin()
        if not binary:
            return ProviderStatus(installed=False)

        # Get version
        version = None
        try:
            r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10)
            version = r.stdout.strip().split("\n")[0] if r.returncode == 0 else None
        except Exception:
            pass

        # Check auth
        authenticated = False
        user = None
        try:
            r = subprocess.run(
                [binary, "auth", "status", "--text"],
                capture_output=True, text=True, timeout=10,
            )
            authenticated = r.returncode == 0
            if authenticated:
                for line in r.stdout.splitlines():
                    if "Email:" in line or "email:" in line.lower():
                        user = line.split(":", 1)[1].strip()
        except Exception:
            pass

        models = [
            ProviderModel(id="sonnet", name="Claude Sonnet", context_window=200_000),
            ProviderModel(id="opus", name="Claude Opus", context_window=200_000),
            ProviderModel(id="haiku", name="Claude Haiku", context_window=200_000),
        ]

        return ProviderStatus(
            installed=True,
            authenticated=authenticated,
            version=version,
            user=user,
            models=models,
        )

    def _build_prompt(self, messages: list[dict]) -> str:
        """Flatten messages into a single prompt string for -p flag."""
        parts = []
        for m in messages:
            role = m["role"].upper()
            parts.append(f"{role}: {m['content']}")
        return "\n\n".join(parts)

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        binary = self._bin()
        if not binary:
            raise RuntimeError("Claude Code CLI not found")

        cfg = get_config()
        model = model or cfg.get("provider.claude_code.model", "sonnet")
        prompt = self._build_prompt(messages)

        result = subprocess.run(
            [
                binary, "-p", prompt,
                "--bare",
                "--output-format", "json",
                "--model", model,
                "--max-turns", "1",
                "--no-session-persistence",
            ],
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Claude Code error: {result.stderr[:500]}")

        try:
            data = json.loads(result.stdout)
            return data.get("result", result.stdout)
        except json.JSONDecodeError:
            return result.stdout.strip()

    def stream(self, messages: list[dict], model: str | None = None) -> Iterator[str]:
        binary = self._bin()
        if not binary:
            raise RuntimeError("Claude Code CLI not found")

        cfg = get_config()
        model = model or cfg.get("provider.claude_code.model", "sonnet")
        prompt = self._build_prompt(messages)

        proc = subprocess.Popen(
            [
                binary, "-p", prompt,
                "--bare",
                "--output-format", "stream-json",
                "--model", model,
                "--max-turns", "1",
                "--no-session-persistence",
                "--verbose",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                # Extract text deltas from stream events
                if event.get("type") == "stream_event":
                    delta = event.get("event", {}).get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")
            except json.JSONDecodeError:
                continue

        proc.wait()

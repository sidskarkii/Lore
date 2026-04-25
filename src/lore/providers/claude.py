"""Claude CLI provider -- uses the host's Claude subscription via subprocess.

Spawns `claude -p` for each completion. Inherits CLAUDE_CODE_OAUTH_TOKEN
from the parent Claude Code process, so no API key is needed.

Sequential execution: one subprocess at a time via a worker queue.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from typing import Iterator

from .base import Provider, ProviderModel, ProviderStatus

_MODEL_MAP = {
    "haiku": "haiku",
    "sonnet": "sonnet",
    "opus": "opus",
}

_TIMEOUT = 120


def _render_messages(messages: list[dict]) -> tuple[str, str]:
    """Split messages into (system_prompt, user_prompt).

    System messages become the --system-prompt value.
    User/assistant messages render as a transcript for -p.
    """
    system_parts = []
    turns = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            turns.append(f"Assistant: {content}")
        else:
            turns.append(f"User: {content}")

    system_prompt = "\n\n".join(system_parts)

    if len(turns) == 1:
        user_prompt = turns[0].removeprefix("User: ")
    else:
        user_prompt = "\n\n".join(turns)

    return system_prompt, user_prompt


class ClaudeProvider(Provider):
    name = "claude"
    display_name = "Claude CLI"

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()

    def detect(self) -> bool:
        has_cli = shutil.which("claude") is not None
        has_token = bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        return has_cli and (has_token or has_key)

    def status(self) -> ProviderStatus:
        if not shutil.which("claude"):
            return ProviderStatus(installed=False, error="claude CLI not found")

        has_auth = bool(
            os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
            or os.environ.get("ANTHROPIC_API_KEY")
        )

        models = [
            ProviderModel(id="haiku", name="Claude Haiku"),
            ProviderModel(id="sonnet", name="Claude Sonnet"),
            ProviderModel(id="opus", name="Claude Opus"),
        ]

        return ProviderStatus(
            installed=True,
            authenticated=has_auth,
            models=models,
        )

    def chat(self, messages: list[dict], model: str | None = None, max_tokens: int = 8192) -> str:
        with self._lock:
            return self._run(messages, model, max_tokens)

    def stream(self, messages: list[dict], model: str | None = None) -> Iterator[str]:
        result = self.chat(messages, model)
        yield result

    def _run(self, messages: list[dict], model: str | None, max_tokens: int) -> str:
        system_prompt, user_prompt = _render_messages(messages)
        resolved_model = _MODEL_MAP.get(model or "", model)

        cmd = ["claude", "-p", "-", "--output-format", "text"]

        if resolved_model:
            cmd.extend(["--model", resolved_model])

        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        print(f"  [claude] chat model={resolved_model or 'default'}")

        try:
            result = subprocess.run(
                cmd,
                input=user_prompt,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
                start_new_session=True,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                if "429" in stderr or "rate limit" in stderr.lower() or "overloaded" in stderr.lower():
                    raise RuntimeError(f"Rate limited: {stderr[:200]}")
                if "not logged in" in stderr.lower() or "oauth" in stderr.lower() or "api key" in stderr.lower():
                    raise RuntimeError(f"Auth error: {stderr[:200]}")
                raise RuntimeError(f"claude CLI failed (exit {result.returncode}): {stderr[:300]}")

            text = result.stdout.strip()
            if not text:
                raise RuntimeError("claude CLI returned empty response")

            return text

        except subprocess.TimeoutExpired:
            print(f"  [claude] timeout after {_TIMEOUT}s")
            raise RuntimeError(f"claude CLI timed out after {_TIMEOUT}s")
        except Exception as e:
            if "Rate limited" in str(e) or "Auth error" in str(e):
                raise
            print(f"  [claude] error: {e}")
            raise

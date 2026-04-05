"""Codex CLI provider — uses OpenAI's `codex` CLI with existing subscription.

Invocation (non-interactive):
    codex exec "{prompt}" --json --full-auto --ephemeral

Output format (--json): JSONL events, one per line.
    - {"type":"item.completed","item":{"type":"agent_message","text":"the answer"}}
    - {"type":"turn.completed","usage":{...}}
    - {"type":"error","message":"..."}

Auth: OAuth via `codex login` or OPENAI_API_KEY env var.
Check: `codex login status`

Note: Don't specify --model unless you know it's available on the account.
      Default model works for ChatGPT subscribers.
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
            ProviderModel(id="default", name="Default (account default)", context_window=128_000),
        ]

        return ProviderStatus(
            installed=True, authenticated=authenticated,
            version=version, models=models,
        )

    def _parse_jsonl(self, output: str) -> str:
        """Extract answer text from Codex JSONL output.

        Answer is in: {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
        Errors in:    {"type":"error","message":"..."}
        """
        parts = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                etype = event.get("type", "")

                if etype == "error":
                    raise RuntimeError(f"Codex error: {event.get('message', 'unknown')}")

                if etype == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        if text:
                            parts.append(text)
            except json.JSONDecodeError:
                continue

        return "\n".join(parts)

    def _build_prompt(self, messages: list[dict]) -> str:
        return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        binary = self._bin()
        if not binary:
            raise RuntimeError("Codex CLI not found")

        prompt = self._build_prompt(messages)

        # Use stdin for long prompts to avoid Windows CLI argument length limits
        result = subprocess.run(
            [binary, "exec", "-", "--json", "--full-auto", "--ephemeral"],
            input=prompt, capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",
        )

        if result.returncode != 0:
            raise RuntimeError(f"Codex error (exit {result.returncode}): {result.stderr[:300]}")

        answer = self._parse_jsonl(result.stdout)
        if not answer:
            # Fallback: try with prompt as argument (short prompts)
            result = subprocess.run(
                [binary, "exec", prompt[:8000], "--json", "--full-auto", "--ephemeral"],
                capture_output=True, text=True, timeout=120,
                encoding="utf-8", errors="replace",
            )
            answer = self._parse_jsonl(result.stdout)

        if not answer:
            raise RuntimeError("Codex returned empty response")
        return answer

    def stream(self, messages: list[dict], model: str | None = None) -> Iterator[str]:
        binary = self._bin()
        if not binary:
            raise RuntimeError("Codex CLI not found")

        prompt = self._build_prompt(messages)

        proc = subprocess.Popen(
            [binary, "exec", "-", "--json", "--full-auto", "--ephemeral"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        proc.stdin.write(prompt)
        proc.stdin.close()

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        if text:
                            yield text
            except json.JSONDecodeError:
                continue

        proc.wait()

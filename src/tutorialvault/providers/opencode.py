"""OpenCode provider — uses the `opencode` CLI with 6 free models.

Invocation (non-interactive):
    opencode run "{prompt}" -m opencode/minimax-m2.5-free --format json

Output format (--format json): JSONL events, one per line.
    - {"type":"text","part":{"text":"the answer"}}
    - {"type":"step_start",...}  — ignore
    - {"type":"step_finish",...} — ignore

Auth: OAuth for Anthropic/OpenAI, or use free opencode/* models with no auth.
Check: `opencode auth list`
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Iterator

from .base import Provider, ProviderModel, ProviderStatus
from ..core.config import get_config

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

        authenticated = False
        try:
            r = subprocess.run(
                [binary, "auth", "list"],
                capture_output=True, text=True, timeout=10,
            )
            authenticated = "credential" in r.stdout.lower() or "oauth" in r.stdout.lower()
        except Exception:
            pass

        return ProviderStatus(
            installed=True, authenticated=authenticated,
            version=version, models=list(_FREE_MODELS),
        )

    def _parse_jsonl(self, output: str) -> str:
        """Extract answer from OpenCode JSONL.

        Answer is in: {"type":"text","part":{"text":"Hello"}}
        """
        parts = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "text":
                    text = event.get("part", {}).get("text", "")
                    if text:
                        parts.append(text)
            except json.JSONDecodeError:
                continue
        return "".join(parts).strip()

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        binary = self._bin()
        if not binary:
            raise RuntimeError("OpenCode CLI not found")

        cfg = get_config()
        model = model or cfg.get("provider.opencode.model", "opencode/minimax-m2.5-free")
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

        result = subprocess.run(
            [binary, "run", prompt, "-m", model, "--format", "json"],
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",
        )

        if result.returncode != 0:
            raise RuntimeError(f"OpenCode error (exit {result.returncode}): {result.stderr[:300]}")

        answer = self._parse_jsonl(result.stdout)
        if not answer:
            raise RuntimeError("OpenCode returned empty response")
        return answer

    def stream(self, messages: list[dict], model: str | None = None) -> Iterator[str]:
        binary = self._bin()
        if not binary:
            raise RuntimeError("OpenCode CLI not found")

        cfg = get_config()
        model = model or cfg.get("provider.opencode.model", "opencode/minimax-m2.5-free")
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

        proc = subprocess.Popen(
            [binary, "run", prompt, "-m", model, "--format", "json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "text":
                    text = event.get("part", {}).get("text", "")
                    if text:
                        yield text
            except json.JSONDecodeError:
                continue

        proc.wait()

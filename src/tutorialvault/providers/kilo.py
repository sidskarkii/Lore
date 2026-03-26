"""Kilo CLI provider — 10 free models, no API key needed.

Invocation:
    kilo --auto --json --yolo -M "kilo-auto/free" --timeout 60 "{prompt}"

Auth: Auto-authenticated on install. `kilo auth` to manage.
Free models: kilo-auto/free, xiaomi/mimo-v2-pro:free, nvidia/nemotron-3-super:free, etc.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Iterator

from .base import Provider, ProviderModel, ProviderStatus
from ..core.config import get_config

# Known free models (verified via `kilo models` output)
_FREE_MODELS = [
    ProviderModel(id="kilo-auto/free", name="Kilo Auto Free", free=True, context_window=204_800),
    ProviderModel(id="xiaomi/mimo-v2-pro:free", name="MiMo V2 Pro (free)", free=True, context_window=1_048_576),
    ProviderModel(id="nvidia/nemotron-3-super-120b-a12b:free", name="Nemotron 3 Super (free)", free=True, context_window=262_144),
    ProviderModel(id="minimax/minimax-m2.5:free", name="MiniMax M2.5 (free)", free=True, context_window=204_800),
    ProviderModel(id="xiaomi/mimo-v2-omni:free", name="MiMo V2 Omni (free)", free=True, context_window=262_144),
    ProviderModel(id="x-ai/grok-code-fast-1:optimized:free", name="Grok Code Fast (free)", free=True, context_window=256_000),
    ProviderModel(id="stepfun/step-3.5-flash:free", name="Step 3.5 Flash (free)", free=True, context_window=256_000),
    ProviderModel(id="arcee-ai/trinity-large-preview:free", name="Trinity Large (free)", free=True, context_window=131_000),
    ProviderModel(id="corethink:free", name="CoreThink (free)", free=True, context_window=78_000),
    ProviderModel(id="openrouter/free", name="OpenRouter Free", free=True, context_window=200_000),
]


class KiloProvider(Provider):
    name = "kilo"
    display_name = "Kilo CLI"
    install_command = "npm install -g @kilocode/cli"

    def _bin(self) -> str | None:
        # kilo CLI registers as both `kilo` and `kilocode`
        return shutil.which("kilo") or shutil.which("kilocode")

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

        # Kilo auto-authenticates, so if installed it's ready
        return ProviderStatus(
            installed=True,
            authenticated=True,
            version=version,
            models=list(_FREE_MODELS),
        )

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        binary = self._bin()
        if not binary:
            raise RuntimeError("Kilo CLI not found")

        cfg = get_config()
        model = model or cfg.get("provider.kilo.model", "kilo-auto/free")
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

        result = subprocess.run(
            [
                binary,
                "--auto", "--json", "--yolo",
                "-M", model,
                "--timeout", "60",
                prompt,
            ],
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Kilo error: {result.stderr[:500]}")

        # Parse JSON output
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
            raise RuntimeError("Kilo CLI not found")

        cfg = get_config()
        model = model or cfg.get("provider.kilo.model", "kilo-auto/free")
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

        proc = subprocess.Popen(
            [
                binary,
                "--auto", "--json", "--yolo",
                "-M", model,
                "--timeout", "60",
                prompt,
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
